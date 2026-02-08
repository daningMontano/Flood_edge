# src/main_stream.py
# ------------------------------------------------------------
# Stream RTSP desde cámara Amcrest, ejecuta segmentación de inundación
# cada N frames, superpone máscara + puntos de severidad y envía
# alertas por Telegram solo cuando cambia la severidad.
# ------------------------------------------------------------

import time
import cv2
from dotenv import load_dotenv
import os
from urllib.parse import quote  # Para codificar usuario/clave RTSP

# --- Módulos internos del proyecto ---
from src.inference.detector import FloodDetectorEdge
from src.preprocess.image import preprocess_image
from src.postprocess.mask import postprocess_mask, overlay_mask_with_points
from src.io.points_loader import load_points_csv
from src.geometry.mask_points import labeled_points_in_mask
from src.actions.telegram_alert import send_alert

# --- Rutas de recursos ---
MODEL_PATH = "models/flood_segmentation_dinov3.onnx"
POINTS_PATH = "data/alert_points/points_video3.csv"


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
# ------------------------------------------------------------
def build_rtsp_url() -> str:
    rtsp_user = os.getenv("RTSP_USER", "")
    rtsp_pass = os.getenv("RTSP_PASS", "")
    rtsp_host = os.getenv("RTSP_HOST", "")
    rtsp_port = os.getenv("RTSP_PORT", "554")

    # Ruta típica Amcrest / Dahua
    rtsp_path = os.getenv(
        "RTSP_PATH",
        "/cam/realmonitor?channel=1&subtype=1"  # substream recomendado
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

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara/stream: {source}")
    return cap


# ------------------------------------------------------------
# Reabre el stream si se cae
# ------------------------------------------------------------
def reopen_capture(source: str, wait_s: float = 0.5) -> cv2.VideoCapture:
    time.sleep(wait_s)
    try:
        return open_capture(source)
    except Exception:
        return None


# ------------------------------------------------------------
# Dibuja TODOS los puntos de severidad del CSV sobre la imagen
# (independiente de si están dentro o fuera de la máscara)
# Sirve para verificar visualmente su ubicación
# ------------------------------------------------------------
def draw_severity_points(image, points):
    h, w = image.shape[:2]

    for p in points:
        if "x" not in p or "y" not in p:
            continue

        # Coordenadas del punto
        x = int(float(p["x"]))
        y = int(float(p["y"]))

        # Evita dibujar fuera del frame
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

    return image


# ------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ------------------------------------------------------------
def main():
    # Carga variables del archivo .env
    load_dotenv()

    # Inicializa modelo de segmentación (ONNX)
    detector = FloodDetectorEdge(MODEL_PATH)

    # Carga puntos de severidad (CSV)
    points = load_points_csv(POINTS_PATH)

    # Construye RTSP y fuerza transporte TCP (más estable)
    source = build_rtsp_url()
    source_tcp = source + ("&rtsp_transport=tcp" if "?" in source else "?rtsp_transport=tcp")

    every_n = 30                # Procesar cada N frames
    last_severity = "bajo"     # Estado previo
    frame_idx = -1

    # Abre stream
    cap = open_capture(source_tcp)

    # Control de reconexión
    fail_reads = 0
    max_fail_reads = 30

    # Ventana de visualización (solo frames procesados)
    window_name = "Prediction Frame (processed only)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()

            # Manejo de caída del stream
            if not ok or frame is None:
                fail_reads += 1
                if fail_reads >= max_fail_reads:
                    cap.release()
                    cap = reopen_capture(source_tcp) or reopen_capture(source)
                    if cap is None:
                        time.sleep(1.0)
                        continue
                    fail_reads = 0
                time.sleep(0.05)
                continue

            fail_reads = 0
            frame_idx += 1

            # SOLO frames que entran al modelo
            if frame_idx % every_n != 0:
                continue

            # ---------------- INFERENCIA ----------------
            input_tensor = preprocess_image(frame)
            output = detector.predict(input_tensor)

            # Mapa de probabilidad → máscara binaria
            prob_map = output[0, 0]
            mask = postprocess_mask(prob_map, frame.shape[:2])

            # Puntos que caen dentro de la máscara
            points_inside = labeled_points_in_mask(mask, points)

            # Superposición máscara + puntos internos
            overlay = overlay_mask_with_points(
                image=frame,
                binary_mask=mask,
                points_all=points,
                points_inside=points_inside
            )

            # Dibuja SIEMPRE todos los puntos del CSV
            overlay = draw_severity_points(overlay, points)

            # Severidad global del frame
            current_severity = get_severity(points_inside)

            # Texto informativo
            cv2.putText(
                overlay,
                f"Severity: {current_severity.upper()} | Frame {frame_idx} | Inside {len(points_inside)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
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
                    try:
                        send_alert(overlay, points_inside)
                    except Exception as e:
                        print(f"[WARN] Telegram falló: {e}")
                last_severity = current_severity

            # Log de depuración
            print(
                f"frame={frame_idx} "
                f"severity={current_severity} "
                f"points_inside={len(points_inside)}"
            )

            # Liberar referencias grandes
            del overlay, mask, output, input_tensor

    finally:
        # Liberación limpia de recursos
        try:
            cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
