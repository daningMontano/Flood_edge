# src/main_video.py
import os
import cv2
from dotenv import load_dotenv

from src.inference.detector import FloodDetectorEdge
from src.preprocess.image import preprocess_image
from src.postprocess.mask import postprocess_mask, overlay_mask_with_points
from src.io.points_loader import load_points_csv
from src.io.video_loader import iter_video_frames
from src.geometry.mask_points import labeled_points_in_mask
from src.actions.telegram_alert import send_alert


MODEL_PATH = "models/flood_segmentation_dinov3.onnx"
POINTS_VIDEO_PATH = "data/alert_points/points_video.csv"
VIDEO_PATH = "data/samples/video_test2.mp4"
OUTPUT_DIR = "data/samples/video_out"


def get_severity(points_inside) -> str:
    """
    Retorna: 'alto', 'medio', 'bajo'
    Regla de prioridad: alto > medio > bajo
    Si no hay puntos dentro, retorna 'bajo' (resetea estado).
    """
    has_medium = False
    has_low = False

    for p in points_inside:
        label = (p.get("label") or p.get("Label") or "").strip().lower()
        if label == "alto":
            return "alto"
        if label == "medio":
            has_medium = True
        if label == "bajo":
            has_low = True

    if has_medium:
        return "medio"
    if has_low:
        return "bajo"
    return "bajo"


def main():
    load_dotenv()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detector = FloodDetectorEdge(MODEL_PATH)
    points = load_points_csv(POINTS_VIDEO_PATH)

    last_severity = "bajo"  # estado anterior observado

    for frame_idx, frame in iter_video_frames(VIDEO_PATH, every_n=3):
        input_tensor = preprocess_image(frame)
        output = detector.predict(input_tensor)

        prob_map = output[0, 0]
        mask = postprocess_mask(prob_map, frame.shape[:2])

        points_inside = labeled_points_in_mask(mask, points)

        overlay = overlay_mask_with_points(
            image=frame,
            binary_mask=mask,
            points_all=points,
            points_inside=points_inside
        )

        out_path = os.path.join(OUTPUT_DIR, f"overlay_{frame_idx}.jpg")
        cv2.imwrite(out_path, overlay)

        current_severity = get_severity(points_inside)

        # Regla: solo enviar si cambia la categoría
        # - Si cambia a medio: enviar una vez
        # - Si sigue en medio: no enviar
        # - Si cambia a alto: enviar una vez
        # - Si sigue en alto: no enviar
        # - Si baja a medio: vuelve a enviar (porque cambió)
        # - Si baja a bajo: no envía, pero actualiza estado (permite re-disparo cuando suba)
        if current_severity != last_severity:
            if current_severity in ("medio", "alto"):
                try:
                    send_alert(out_path, points_inside)  # caption automático: "alerta media" o "alerta alta"
                except Exception as e:
                    print(f"[WARN] Telegram falló: {e}")

            last_severity = current_severity

        # Log mínimo por frame procesado
        print(f"frame={frame_idx} severity={current_severity} points_inside={len(points_inside)}")


if __name__ == "__main__":
    main()
