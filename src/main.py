from dotenv import load_dotenv

from src.inference.detector import FloodDetectorEdge
from src.preprocess.image import preprocess_image
from src.postprocess.mask import (
    postprocess_mask,
    overlay_mask_with_points
)
from src.io.image_loader import load_image
from src.io.points_loader import load_points_csv
from src.geometry.mask_points import labeled_points_in_mask
from src.actions.telegram_alert import send_alert

MODEL_PATH = "models/flood_segmentation_dinov3.onnx"
POINTS_PATH = "data/alert_points/points.csv"


def main(image_path: str):
    # 1) Variables de entorno
    load_dotenv()

    # 2) Modelo
    detector = FloodDetectorEdge(MODEL_PATH)

    # 3) Imagen
    image = load_image(image_path)

    # 4) Preprocesamiento
    input_tensor = preprocess_image(image)

    # 5) Inferencia
    output = detector.predict(input_tensor)

    # 6) Máscara
    prob_map = output[0, 0]
    mask = postprocess_mask(prob_map, image.shape[:2])

    # 7) Puntos
    points = load_points_csv(POINTS_PATH)
    points_inside = labeled_points_in_mask(mask, points)

    # 8) Overlay (solo en RAM)
    result = overlay_mask_with_points(
        image=image,
        binary_mask=mask,
        points_all=points,
        points_inside=points_inside
    )

    # 9) Acción (sin archivos)
    try:
        send_alert(result, points_inside)
    except Exception as e:
        print(f"[WARN] Telegram falló: {e}")

    # 10) Log
    if points_inside:
        print("Puntos dentro de la inundación:")
        for p in points_inside:
            print(f"- {p['label']} ({p['x']}, {p['y']})")
    else:
        print("Sin puntos dentro de la inundación")


if __name__ == "__main__":
    main("data/samples/test2.jpeg")
