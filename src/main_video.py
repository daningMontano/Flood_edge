# src/main_video.py
import os
from dotenv import load_dotenv

from src.inference.detector import FloodDetectorEdge
from src.preprocess.image import preprocess_image
from src.postprocess.mask import postprocess_mask, overlay_mask_with_points
from src.io.points_loader import load_points_csv
from src.io.video_loader import iter_video_frames
from src.geometry.mask_points import labeled_points_in_mask
from src.actions.telegram_alert import send_alert


MODEL_PATH = "models/flood_segmentation_dinov3.onnx"
POINTS_VIDEO_PATH = "data/alert_points/points_video4.csv"
VIDEO_PATH = "data/samples/video_test3.mp4"


def get_severity(points_inside) -> str:
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

    detector = FloodDetectorEdge(MODEL_PATH)
    points = load_points_csv(POINTS_VIDEO_PATH)

    last_severity = "bajo"

    for frame_idx, frame in iter_video_frames(VIDEO_PATH, every_n=30):  # <- 3 si esa es tu regla
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

        current_severity = get_severity(points_inside)

        # Solo enviar si cambia y es medio/alto
        if current_severity != last_severity:
            if current_severity in ("medio", "alto"):
                try:
                    send_alert(overlay, points_inside)  # <- imagen en RAM
                except Exception as e:
                    print(f"[WARN] Telegram falló: {e}")

            last_severity = current_severity

        print(f"frame={frame_idx} severity={current_severity} points_inside={len(points_inside)}")

        # liberar referencias grandes (opcional)
        del overlay, mask, output, input_tensor


if __name__ == "__main__":
    main()

