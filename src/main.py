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

import cv2

MODEL_PATH = "models/flood_segmentation_dinov3.onnx"
POINTS_PATH = "data/alert_points/points.csv"


def main(image_path: str, output_path: str):
    # Cargar variables desde .env
    load_dotenv()

    # 1) Inicializar modelo
    detector = FloodDetectorEdge(MODEL_PATH)

    # 2) Cargar imagen
    image = load_image(image_path)

    # 3) Preprocesar
    input_tensor = preprocess_image(image)

    # 4) Inferencia
    output = detector.predict(input_tensor)

    # 5) Postprocesar máscara
    prob_map = output[0, 0]          # (H, W)
    mask = postprocess_mask(prob_map, image.shape[:2])

    # 6) Cargar puntos desde CSV
    points = load_points_csv(POINTS_PATH)

    # 7) Puntos dentro de la máscara
    points_inside = labeled_points_in_mask(mask, points)

    # 8) Overlay (imagen + máscara + puntos)
    result = overlay_mask_with_points(
        image=image,
        binary_mask=mask,
        points_all=points,
        points_inside=points_inside
    )

    # 9) Guardar salida
    cv2.imwrite(output_path, result)

    # 10) Acción: enviar alerta si corresponde (ALTO o MEDIO)
    try:
        send_alert(output_path, points_inside)
    except Exception as e:
        print(f"[WARN] Telegram falló: {e}")

    # 11) Log mínimo
    print("Puntos dentro de la inundación:")
    for p in points_inside:
        label = p.get("label") or p.get("Label") or ""
        x = p.get("x") or p.get("X")
        y = p.get("y") or p.get("Y")
        print(f"- {label} ({x}, {y})")


if __name__ == "__main__":
    main(
        image_path="data/samples/test2.jpeg",
        output_path="data/samples/output2.jpg"
    )
