import csv
from typing import List, Dict

def load_points_csv(path: str) -> List[Dict]:
    points = []
    with open(path, newline="", encoding="utf-8-sig") as f:  # <-- cambio clave
        reader = csv.DictReader(f)
        for row in reader:
            points.append({
                "label": row["Label"].strip(),
                "x": int(row["X"]),
                "y": int(row["Y"])
            })
    return points
