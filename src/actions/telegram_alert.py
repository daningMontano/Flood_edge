# src/actions/telegram_alert.py
import os
import cv2
import requests
import numpy as np
from typing import Iterable, Dict, Any, Optional, Union


def _get_severity(points_inside: Iterable[Dict[str, Any]]) -> Optional[str]:
    has_medium = False
    for p in points_inside:
        label = (p.get("label") or p.get("Label") or "").strip().lower()
        if label == "alto":
            return "alto"
        if label == "medio":
            has_medium = True
    return "medio" if has_medium else None


def send_alert(image: Union[str, np.ndarray], points_inside: Iterable[Dict[str, Any]]) -> None:
    """
    Acepta:
      - str: ruta a archivo (compatibilidad hacia atrás)
      - np.ndarray: imagen BGR en memoria (no escribe a disco)
    """
    severity = _get_severity(points_inside)
    if severity is None:
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Variables de entorno TELEGRAM_* no definidas")

    caption = "alerta alta" if severity == "alto" else "alerta media"
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    # ---- Caso 1: image es path (legacy) ----
    if isinstance(image, str):
        with open(image, "rb") as img:
            r = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": img},
                timeout=30
            )
        if not r.ok:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")
        r.raise_for_status()
        return

    # ---- Caso 2: image es np.ndarray (RAM) ----
    if not isinstance(image, np.ndarray):
        raise TypeError("image debe ser str (path) o np.ndarray (imagen BGR)")

    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen a JPG")

    files = {"photo": ("alert.jpg", buffer.tobytes(), "image/jpeg")}
    r = requests.post(
        url,
        data={"chat_id": chat_id, "caption": caption},
        files=files,
        timeout=30
    )

    if not r.ok:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")
    r.raise_for_status()
