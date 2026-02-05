from typing import List, Dict
import numpy as np

def labeled_points_in_mask(
    mask: np.ndarray,
    points: List[Dict],
    *,
    inside_value: int = 1
) -> List[Dict]:
    h, w = mask.shape[:2]
    selected = []

    for p in points:
        x, y = int(p["x"]), int(p["y"])
        if 0 <= x < w and 0 <= y < h:
            if int(mask[y, x]) == inside_value:
                selected.append(p)

    return selected
