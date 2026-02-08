# src/main_stream.py
import time
import cv2
from dotenv import load_dotenv
import os
from urllib.parse import quote

from src.inference.detector import FloodDetectorEdge
from src.preprocess.image import preprocess_image
from src.postprocess.mask import postprocess_mask, overlay_mask_with_points
from src.io.points_loader import load_points_csv
from src.geometry.mask_points import labeled_points_in_mask
from src.actions.telegram_alert import send_alert

MODEL_PATH = "models/flood_segmentation_dinov3.onnx"
POINTS_PATH = "data/alert_points/points_video4.csv"


def get_severity(points_inside) -> str:
    has_medium = False
    for p in points_inside:
        label = (p.get("label") or p.get("Label") or "").strip().lower()
        if label == "alto":
            return "alto"
        if label == "medio":
            has_medium = True
    return "medio" if has_medium else "bajo"


def build_rtsp_url() -> str:
    rtsp_user = os.getenv("RTSP_USER", "")
    rtsp_pass = os.getenv("RTSP_PASS", "")
    rtsp_host = os.getenv("RTSP_HOST", "")
    rtsp_port = os.getenv("RTSP_PORT", "554")
    rtsp_path = os.getenv("RTSP_PATH", "/cam/realmonitor?channel=1&subtype=1")

    user_enc = quote(rtsp_user, safe="")
    pass_enc = quote(rtsp_pass, safe="")

    if user_enc and pass_enc:
        return f"rtsp://{user_enc}:{pass_enc}@{rtsp_host}:{rtsp_port}{rtsp_path}"
    return f"rtsp://{rtsp_host}:{rtsp_port}{rtsp_path}"


def open_capture(source: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el stream: {source}")
    return cap


def _label_norm(label: str) -> str:
    return (label or "").strip().lower()


def draw_severity_points(image, points):
    """
    Dibuja TODOS los puntos (Bajo/Medio/Alto) siempre, para ver ubicación.
    Espera puntos con llaves: x, y, label (o Label).
    """
    for p in points:
        x, y = int(p["x"]), int(p["y"])
        label = _label_norm(p.get("label") or p.get("Label"))

        # Colores BGR
        if label == "alto":
            color = (0, 0, 255)      # rojo
            txt = "ALTO"
        elif label == "medio":
            color = (0, 255, 255)    # amarillo
            txt = "MEDIO"
        else:
            color = (0, 255, 0)      # verde
            txt = "BAJO"

        # Punto + borde
        cv2.circle(image, (x, y), 7, color, -1)
        cv2.circle(image, (x, y), 9, (0, 0, 0), 2)

        # Etiqueta con fondo para legibilidad
        label_text = f"{txt} ({x},{y})"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x0, y0 = x + 12, y - 12
        cv2.rectangle(image, (x0, y0 - th - 6), (x0 + tw + 6, y0 + 6), (0, 0, 0), -1)
        cv2.putText(image, label_text, (x0 + 3, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def main():
    load_dotenv()

    detector = FloodDetectorEdge(MODEL_PATH)
    points = load_points_csv(POINTS_PATH)

    source = build_rtsp_url()
    source = source + ("&rtsp_transport=tcp" if "?" in source else "?rtsp_transport=tcp")

    every_n = 3
    last_severity = "bajo"
    frame_idx = -1

    cap = open_capture(source)

    cv2.namedWindow("Prediction Frame", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame_idx += 1

            # SOLO frames que entran al modelo
            if frame_idx % every_n != 0:
                continue

            # --- INFERENCIA ---
            input_tensor = preprocess_image(frame)
            output = detector.predict(input_tensor)

            prob_map = output[0, 0]
            mask = postprocess_mask(prob_map, frame.shape[:2])

            points_inside = labeled_points_in_mask(mask, points)
            current_severity = get_severity(points_inside)

            overlay = overlay_mask_with_points(
                image=frame,
                binary_mask=mask,
                points_all=points,
                points_inside=points_inside
            )

            # ✅ Dibuja TODOS los puntos de severidad (ubicación fija)
            overlay = draw_severity_points(overlay, points)

            # Texto de estado
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

            cv2.imshow("Prediction Frame", overlay)

            # ESC para salir
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Alertas solo por cambio
            if current_severity != last_severity:
                if current_severity in ("medio", "alto"):
                    send_alert(overlay, points_inside)
                last_severity = current_severity

            print(f"frame={frame_idx} severity={current_severity} inside={len(points_inside)}")

            del overlay, mask, output, input_tensor

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
