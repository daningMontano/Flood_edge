# ------------------------------------------------------------
# Stream RTSP (headless, sin GUI):
# - Lee frames desde una cámara RTSP.
# - Ejecuta segmentación cada N frames (STREAM_EVERY_N).
# - Calcula severidad según puntos dentro de la máscara.
# - Envía alerta por Telegram solo cuando cambia la severidad (medio/alto).
# - Guarda métricas en CSV por cada inferencia: CPU, RAM, tiempo de inferencia.
#
# Importante:
# - NO usa cv2.imshow / cv2.waitKey / cv2.destroyAllWindows.
# - Funciona con opencv-python-headless.
# ------------------------------------------------------------

import os
import csv
import time
from datetime import datetime, timezone
from urllib.parse import quote  # URL-encode user/pass para RTSP

import cv2
from dotenv import load_dotenv

# psutil obligatorio para métricas CPU/RAM
try:
    import psutil
except ImportError as e:
    raise ImportError(
        "Falta psutil. Instala con: pip install psutil (o agrégalo a tu environment.yml)"
    ) from e

from src.inference.detector import FloodDetectorEdge
from src.preprocess.image import preprocess_image
from src.postprocess.mask import postprocess_mask, overlay_mask_with_points
from src.io.points_loader import load_points_csv
from src.geometry.mask_points import labeled_points_in_mask
from src.actions.telegram_alert import send_alert


# --- Defaults (puedes sobreescribir por variables de entorno) ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/flood_segmentation_dinov3.onnx")
POINTS_PATH = os.getenv("POINTS_PATH", "data/alert_points/points_video4.csv")

# Procesa 1 de cada N frames (si no existe variable de entorno, usa 30)
STREAM_EVERY_N = int(os.getenv("STREAM_EVERY_N", "150"))

# CSV métricas
METRICS_CSV_PATH = os.getenv("METRICS_CSV_PATH", "metrics/metrics_stream.csv")

# Reintentos de lectura antes de reconectar
STREAM_MAX_FAIL_READS = int(os.getenv("STREAM_MAX_FAIL_READS", "30"))


# ------------------------------------------------------------
# Severidad global a partir de puntos dentro de la máscara
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
# Construye URL RTSP desde .env:
# RTSP_USER, RTSP_PASS, RTSP_HOST, RTSP_PORT, RTSP_PATH
#
# Nota:
# - subtype=0 (main stream) suele coincidir con CSV de puntos en HD/4K.
# - subtype=1 (substream) suele ser baja resolución: puntos podrían quedar fuera.
# ------------------------------------------------------------
def build_rtsp_url() -> str:
    rtsp_user = os.getenv("RTSP_USER", "")
    rtsp_pass = os.getenv("RTSP_PASS", "")
    rtsp_host = os.getenv("RTSP_HOST", "")
    rtsp_port = os.getenv("RTSP_PORT", "554")
    rtsp_path = os.getenv("RTSP_PATH", "/cam/realmonitor?channel=1&subtype=0")

    if not rtsp_host:
        raise RuntimeError("RTSP_HOST no definido (revisa tu .env)")

    if rtsp_path and not rtsp_path.startswith("/"):
        rtsp_path = "/" + rtsp_path

    user_enc = quote(rtsp_user, safe="")
    pass_enc = quote(rtsp_pass, safe="")

    if user_enc and pass_enc:
        return f"rtsp://{user_enc}:{pass_enc}@{rtsp_host}:{rtsp_port}{rtsp_path}"
    return f"rtsp://{rtsp_host}:{rtsp_port}{rtsp_path}"


# ------------------------------------------------------------
# Abre RTSP con FFmpeg (más estable). Reduce buffer para latencia.
# ------------------------------------------------------------
def open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # En algunos builds ayuda a forzar conversión a BGR
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el stream: {source}")
    return cap


def reopen_capture(source: str, wait_s: float = 0.8) -> cv2.VideoCapture | None:
    time.sleep(wait_s)
    try:
        return open_capture(source)
    except Exception:
        return None


# ------------------------------------------------------------
# Asegura que el frame sea BGR (3 canales) para overlays correctos.
# Si llega GRAY/BGRA, convierte a BGR.
# ------------------------------------------------------------
def ensure_bgr(frame):
    if frame is None:
        return None

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    return frame


# ------------------------------------------------------------
# Dibuja todos los puntos del CSV sobre la imagen (solo para depurar
# por si quieres enviar imagen con puntos SIEMPRE visibles).
# NO reescala puntos: asume que coinciden con la resolución del stream.
# ------------------------------------------------------------
def draw_severity_points(image, points):
    h, w = image.shape[:2]

    for p in points:
        if "x" not in p or "y" not in p:
            continue

        x = int(float(p["x"]))
        y = int(float(p["y"]))

        # Si los puntos no coinciden con la resolución, quedarán fuera y no se dibujan
        if not (0 <= x < w and 0 <= y < h):
            continue

        label = (p.get("label") or p.get("Label") or "").strip().lower()
        if label == "alto":
            color, txt = (0, 0, 255), "ALTO"
        elif label == "medio":
            color, txt = (0, 255, 255), "MEDIO"
        else:
            color, txt = (0, 255, 0), "BAJO"

        # Punto grande para HD/4K
        cv2.circle(image, (x, y), 6, color, -1)
        cv2.circle(image, (x, y), 8, (0, 0, 0), 3)

        # Etiqueta
        cv2.putText(
            image, txt, (x + 25, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 2 ,cv2.LINE_AA
        )
        cv2.putText(
            image, txt, (x + 25, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    return image


# ------------------------------------------------------------
# Inicializa el CSV de métricas (append).
# - Crea directorio si no existe.
# - Escribe header si el archivo no existe.
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
            "source",
            "frame_idx",
            "severity",
            "points_inside",
            "inference_ms",
            "system_cpu_percent",
            "process_cpu_percent",
            "process_rss_mb",
            "system_ram_percent",
            "alert_sent",
            "alert_error",
        ])
        f.flush()

    return f, writer


# ------------------------------------------------------------
# Main (headless)
# ------------------------------------------------------------
def main():
    load_dotenv()

    # Modelo + puntos
    detector = FloodDetectorEdge(MODEL_PATH)
    points = load_points_csv(POINTS_PATH)

    # Métricas (psutil)
    proc = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)     # warm-up
    proc.cpu_percent(interval=None)       # warm-up

    # CSV métricas
    metrics_file, metrics_writer = init_metrics_csv(METRICS_CSV_PATH)

    # RTSP (intenta TCP para estabilidad)
    source = build_rtsp_url()
    source_tcp = source + ("&rtsp_transport=tcp" if "?" in source else "?rtsp_transport=tcp")

    # Abre stream (TCP primero)
    cap = reopen_capture(source_tcp, wait_s=0.0) or reopen_capture(source, wait_s=0.0)
    if cap is None:
        raise RuntimeError("No se pudo abrir el stream RTSP (ni con TCP ni sin TCP).")

    # Estado de alertas (solo dispara si cambia)
    last_severity = "bajo"

    # Control de reconexión
    fail_reads = 0
    frame_idx = -1

    try:
        while True:
            # Si el stream se cayó, reabre
            if cap is None or not cap.isOpened():
                cap = reopen_capture(source_tcp) or reopen_capture(source)
                if cap is None:
                    time.sleep(1.0)
                    continue

            ok, frame = cap.read()

            if not ok or frame is None:
                fail_reads += 1
                if fail_reads >= STREAM_MAX_FAIL_READS:
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

            # Procesar solo cada N frames
            if frame_idx % STREAM_EVERY_N != 0:
                continue

            # Asegura BGR (si el stream viene en gris, lo convertimos a BGR)
            frame = ensure_bgr(frame)

            # ---------------- INFERENCIA (timed) ----------------
            input_tensor = preprocess_image(frame)

            t0 = time.perf_counter()
            output = detector.predict(input_tensor)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # ---------------- Postproceso ----------------
            prob_map = output[0, 0]
            mask = postprocess_mask(prob_map, frame.shape[:2])

            points_inside = labeled_points_in_mask(mask, points)
            current_severity = get_severity(points_inside)

            # Overlay para enviar por Telegram (mask + puntos)
            overlay = overlay_mask_with_points(
                image=frame,
                binary_mask=mask,
                points_all=points,
                points_inside=points_inside,
                radius = 2
            )

            # Opcional: fuerza que SIEMPRE se vean puntos (si caen dentro del frame)
            overlay = draw_severity_points(overlay, points)

            # ---------------- Alertas (solo si cambia) ----------------
            alert_sent = 0
            alert_error = ""
            if current_severity != last_severity:
                if current_severity in ("medio", "alto"):
                    try:
                        send_alert(overlay, points_inside)
                        alert_sent = 1
                    except Exception as e:
                        # No imprimir; registrar en CSV
                        alert_error = str(e)[:300]
                last_severity = current_severity

            # ---------------- Métricas CPU/RAM ----------------
            system_cpu = psutil.cpu_percent(interval=None)
            process_cpu = proc.cpu_percent(interval=None)
            process_rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
            system_ram_percent = psutil.virtual_memory().percent

            # ---------------- Guardar fila en CSV ----------------
            metrics_writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                "stream",
                frame_idx,
                current_severity,
                len(points_inside),
                round(inference_ms, 3),
                round(system_cpu, 2),
                round(process_cpu, 2),
                round(process_rss_mb, 3),
                round(system_ram_percent, 2),
                alert_sent,
                alert_error,
            ])
            metrics_file.flush()

            # Limpieza explícita (útil en edge)
            del overlay, mask, output, input_tensor

    finally:
        try:
            metrics_file.close()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()