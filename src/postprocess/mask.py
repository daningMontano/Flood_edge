import cv2
import numpy as np
from typing import Iterable, Tuple, Optional

Point = Tuple[int, int]  # (x, y)


def postprocess_mask(prob_map, original_shape, threshold=0.5):
    """
    Convierte el mapa de probabilidades en una máscara binaria 0/1.
    """
    h, w = original_shape
    mask = cv2.resize(prob_map, (w, h))
    return (mask > threshold).astype(np.uint8)


def overlay_mask(image, binary_mask, alpha=0.5):
    """
    Superpone la máscara binaria sobre la imagen original (color azul).
    """
    colored = np.zeros_like(image)
    colored[:, :, 0] = binary_mask * 255  # canal azul (BGR)
    return cv2.addWeighted(image, 1, colored, alpha, 0)



def overlay_mask_with_points(
    image: np.ndarray,
    binary_mask: np.ndarray,
    points_all: Iterable[dict],
    points_inside: Optional[Iterable[dict]] = None,
    radius: int = 9
) -> np.ndarray:
    """
    Superpone la máscara y dibuja puntos según nivel (siempre mismo color):
    - Alto  -> rojo
    - Medio -> amarillo
    - Bajo  -> verde

    Los puntos dentro de la máscara se resaltan con un radio mayor.
    """

    COLOR_MAP = {
        "alto":  (0, 0, 255),    # rojo
        "medio": (0, 255, 255),  # amarillo
        "bajo":  (0, 255, 0),    # verde
    }

    # Overlay de la máscara
    out = overlay_mask(image, binary_mask)

    # Conjunto para saber cuáles están dentro
    inside_set = set()
    if points_inside:
        inside_set = {(p["x"], p["y"]) for p in points_inside}

    # Dibujar todos los puntos con color por label
    for p in points_all:
        x = int(p["x"])
        y = int(p["y"])
        label = p["label"].strip().lower()
        color = COLOR_MAP.get(label, (255, 255, 255))

        # Resaltar si está dentro de la máscara
        r = radius + 3 if (x, y) in inside_set else radius

        cv2.circle(out, (x, y), r, color, -1)

    return out
