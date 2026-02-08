# src/main_stream.py
# ------------------------------------------------------------
# Stream RTSP desde cámara Amcrest.
# - Ejecuta segmentación de inundación cada N frames.
# - Superpone máscara + puntos de severidad (siempre visibles).
# - Envía alertas por Telegram solo cuando cambia la severidad.
# - Guarda métricas por frame procesado en CSV: CPU, RAM, tiempo de inferencia.
# ------------------------------------------------------------

import os
import csv
import time
from datetime import datetime, timezone
from urllib.parse import quote  # Para codificar usuario/clave RTSP

import cv2
from dotenv import load_dotenv

# psutil es obligatorio para métricas CPU/RAM
try:
    import psutil
except ImportError as e:
    raise ImportError(
        "Falta psutil. Instala con: pip install psutil (o agrégalo a tu environment.yml)"
    ) from e

# --- Módulos internos del proyecto ---
from src.inference.detector import FloodDetectorEdge
from src.preprocess.image import preprocess_image
from src.postprocess.mask import postprocess_mask, overlay_mask_with_points
from src.io.points_loader import load_points_csv
from src.geometry.mask_points import labeled_points_in_mask
from src.actions.telegram_alert import send_alert

# --- Rutas de recursos ---
MODEL_PATH = "models/flood_segmentation_dinov3.onnx"
POINTS_PATH = "data/alert_points/points_video4.csv"

# --- Métricas (CSV) ---
DEFAULT_METRICS_CSV_PATH = "metrics/metrics_stream.csv"


# ------------------------------------------------------------
# Determina severidad global en función de los puntos dentro
# de la máscara de agua
# Prioridad: ALTO > MEDIO > BAJO
# ------------------------------------------------------------
def get_severity(points_inside) -> str:
    has_medium = False
    for p in points_inside:
        label = (p.get("label") or p.get("Label") or "").strip().lower()
        if label == "alto":
            return "alto"
        if label == "medio":
            has_medium = True
    return "medio" if has_medium else "bajo"


# ------------------------------------------------------------
# Construye URL RTSP desde variables de entorno (.env)
# Incluye URL-encoding para evitar errores por caracteres
# especiales en usuario/contraseña
#
# Nota crítica para tus puntos:
# - Si tus puntos (CSV) fueron definidos sobre el stream principal (HD/4K),
#   NO uses subtype=1 (substream) porque cambia la resolución y los puntos quedarán fuera.
# - Por eso el default aquí usa subtype=0 (main stream).
# ------------------------------------------------------------
def build_rtsp_url() -> str:
    rtsp_user = os.getenv("RTSP_USER", "")
    rtsp_pass = os.getenv("RTSP_PASS", "")
    rtsp_host = os.getenv("RTSP_HOST", "")
    rtsp_port = os.getenv("RTSP_PORT", "554")

    # Ruta típica Amcrest / Dahua
    rtsp_path = os.getenv(
        "RTSP_PATH",
        "/cam/realmonitor?channel=1&subtype=0"  # main stream (más probable que coincida con tus puntos)
    )

    user_enc = quote(rtsp_user, safe="")
    pass_enc = quote(rtsp_pass, safe="")

    if user_enc and pass_enc:
        return f"rtsp://{user_enc}:{pass_enc}@{rtsp_host}:{rtsp_port}{rtsp_path}"
    return f"rtsp://{rtsp_host}:{rtsp_port}{rtsp_path}"


# ------------------------------------------------------------
# Abre el stream RTSP usando FFmpeg (mejor estabilidad)
# ------------------------------------------------------------
def open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    # Reduce latencia interna del buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # En algunos builds ayuda a forzar conversión a BGR
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara/stream: {source}")
    return cap


# ------------------------------------------------------------
# Reabre el stream si se cae
# ------------------------------------------------------------
def reopen_capture(source: str, wait_s: float = 0.8) -> cv2.VideoCapture | None:
    time.sleep(wait_s)
    try:
        return open_capture(source)
    except Exception:
        return None


# ------------------------------------------------------------
# Asegura que el frame sea BGR (3 canales).
# Si el stream llega en escala de grises, lo convierte a BGR para
# que el overlay y los colores de puntos funcionen correctamente.
# ------------------------------------------------------------
def ensure_bgr(frame):
    if frame is None:
        return None

    # Grayscale (H, W)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # (H, W, 1)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # (H, W, 4) -> BGRA to BGR
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


# ------------------------------------------------------------
# Dibuja TODOS los puntos de severidad del CSV sobre la imagen
# (independiente de si están dentro o fuera de la máscara)
# Sirve para verificar visualmente su ubicación.
#
# Importante: NO se reescalan puntos aquí.
# Los puntos se asumen en coordenadas de píxel del mismo tamaño
# que el frame del stream.
# ------------------------------------------------------------
def draw_severity_points(image, points):
    h, w = image.shape[:2]
    drawn = 0

    for p in points:
        if "x" not in p or "y" not in p:
            continue

        x = int(p["x"])
        y = int(p["y"])

        # Evita dibujar fuera del frame (si no coincide la resolución)
        if not (0 <= x < w and 0 <= y < h):
            continue

        label = (p.get("label") or p.get("Label") or "").strip().lower()

        # Colores BGR por severidad
        if label == "alto":
            color, txt = (0, 0, 255), "ALTO"
        elif label == "medio":
            color, txt = (0, 255, 255), "MEDIO"
        else:
            color, txt = (0, 255, 0), "BAJO"

        # Punto visible (radio grande para HD / 4K)
        cv2.circle(image, (x, y), 18, color, -1)
        cv2.circle(image, (x, y), 22, (0, 0, 0), 3)

        # Etiqueta textual
        cv2.putText(
            image, txt, (x + 25, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            image, txt, (x + 25, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        drawn += 1

    return image, drawn


# ------------------------------------------------------------
# Inicializa el CSV de métricas:
# - Crea el directorio si no existe
# - Escribe header si el archivo no existe
# ------------------------------------------------------------
def init_metrics_csv(path: str):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(path)
    f = open(path, mode="a", newline="", encoding="utf-8")
    writer = csv.writer(f)

    if not file_exists:
        writer.writerow([
            "timestamp_utc",
            "frame_idx",
            "severity",
            "points_inside",
            "inference_ms",
            "cpu_percent",
            "process_rss_mb",
            "system_ram_percent",
        ])
        f.flush()

    return f, writer


# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ------------------------------------------------------------
def main():
    # Carga variables del archivo .env
    load_dotenv()

    # Ruta del CSV de métricas (configurable por env)
    metrics_csv_path = os.getenv("METRICS_CSV_PATH", DEFAULT_METRICS_CSV_PATH)

    # Inicializa métricas
    process = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)  # warm-up para evitar 0.0 al primer registro
    metrics_file, metrics_writer = init_metrics_csv(metrics_csv_path)

    # Inicializa modelo de segmentación (ONNX)
    detector = FloodDetectorEdge(MODEL_PATH)

    # Carga puntos de severidad (CSV)
    points = load_points_csv(POINTS_PATH)

    # Construye RTSP y fuerza transporte TCP (más estable)
    source = build_rtsp_url()
    source_tcp = source + ("&rtsp_transport=tcp" if "?" in source else "?rtsp_transport=tcp")

    every_n = int(os.getenv("STREAM_EVERY_N", "30"))  # Procesar cada N frames
    last_severity = "bajo"  # Estado previo
    frame_idx = -1

    # Abre stream
    cap = reopen_capture(source_tcp, wait_s=0.0) or reopen_capture(source, wait_s=0.0)
    if cap is None:
        raise RuntimeError("No se pudo abrir el stream RTSP (ni con TCP ni sin TCP).")

    # Control de reconexión
    fail_reads = 0
    max_fail_reads = int(os.getenv("STREAM_MAX_FAIL_READS", "30"))

    # Ventana de visualización (solo frames procesados)
    window_name = "Prediction Frame (processed only)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Para mostrar un warning si los puntos no coinciden con la resolución
    warned_points_mismatch = False

    try:
        while True:
            # Si el stream se cayó y cap quedó inválido, reintenta abrir
            if cap is None or not cap.isOpened():
                cap = reopen_capture(source_tcp) or reopen_capture(source)
                if cap is None:
                    time.sleep(1.0)
                    continue

            ok, frame = cap.read()

            # Manejo de caída del stream
            if not ok or frame is None:
                fail_reads += 1
                if fail_reads >= max_fail_reads:
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = reopen_capture(source_tcp) or reopen_capture(source)
                    fail_reads = 0
                time.sleep(0.05)
                continue

            fail_reads = 0
            frame_idx += 1

            # SOLO frames que entran al modelo
            if frame_idx % every_n != 0:
                continue

            # Asegura frame BGR (evita overlays que fallen o se vean mal)
            frame = ensure_bgr(frame)

            # ---------------- INFERENCIA (timed) ----------------
            t0 = time.perf_counter()

            input_tensor = preprocess_image(frame)
            output = detector.predict(input_tensor)

            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000.0

            # Mapa de probabilidad → máscara binaria
            prob_map = output[0, 0]
            mask = postprocess_mask(prob_map, frame.shape[:2])

            # Puntos que caen dentro de la máscara
            points_inside = labeled_points_in_mask(mask, points)

            # Superposición máscara + puntos internos (color por severidad)
            overlay = overlay_mask_with_points(
                image=frame,
                binary_mask=mask,
                points_all=points,
                points_inside=points_inside
            )

            # Dibuja SIEMPRE todos los puntos del CSV (para ver su ubicación)
            overlay, drawn_points = draw_severity_points(overlay, points)

            # Severidad global del frame
            current_severity = get_severity(points_inside)

            # Texto informativo
            cv2.putText(
                overlay,
                f"Severity: {current_severity.upper()} | Frame {frame_idx} | Inside {len(points_inside)} | Drawn {drawn_points}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Warning visual si NO se dibuja ningún punto (típicamente por resolución distinta)
            if (not warned_points_mismatch) and drawn_points == 0 and len(points) > 0:
                warned_points_mismatch = True
                cv2.putText(
                    overlay,
                    "WARNING: 0 points drawn. Check RTSP subtype/resolution vs points CSV.",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

            # Mostrar SOLO el frame procesado
            cv2.imshow(window_name, overlay)

            # Salir con ESC
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

            # Enviar alerta solo si cambia la severidad
            if current_severity != last_severity:
                if current_severity in ("medio", "alto"):
                    # send_alert acepta np.ndarray (imagen BGR en memoria)
                    send_alert(overlay, points_inside)
                last_severity = current_severity

            # ---------------- MÉTRICAS -> CSV ----------------
            timestamp_utc = datetime.now(timezone.utc).isoformat()

            # CPU: porcentaje desde la última llamada (aprox. durante este ciclo)
            cpu_percent = psutil.cpu_percent(interval=None)

            # RAM:
            process_rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
            system_ram_percent = psutil.virtual_memory().percent

            metrics_writer.writerow([
                timestamp_utc,
                frame_idx,
                current_severity,
                len(points_inside),
                round(inference_ms, 3),
                round(cpu_percent, 2),
                round(process_rss_mb, 2),
                round(system_ram_percent, 2),
            ])
            metrics_file.flush()

            # Liberar referencias grandes
            del overlay, mask, output, input_tensor

    finally:
        # Liberación limpia de recursos
        try:
            metrics_file.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
