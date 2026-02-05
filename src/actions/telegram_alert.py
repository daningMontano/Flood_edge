# src/actions/telegram_alert.py
import os
import requests
from typing import Iterable, Dict, Any, Optional


def _get_severity(points_inside: Iterable[Dict[str, Any]]) -> Optional[str]:
    has_medium = False
    for p in points_inside:
        label = (p.get("label") or p.get("Label") or "").strip().lower()
        if label == "alto":
            return "alto"
        if label == "medio":
            has_medium = True
    return "medio" if has_medium else None


def send_alert(image_path: str, points_inside: Iterable[Dict[str, Any]]) -> None:
    severity = _get_severity(points_inside)
    if severity is None:
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Variables de entorno TELEGRAM_* no definidas")

    caption = "alerta alta" if severity == "alto" else "alerta media"
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    with open(image_path, "rb") as img:
        r = requests.post(
            url,
            data={"chat_id": chat_id, "caption": caption},
            files={"photo": img},
            timeout=30
        )

    if not r.ok:
        raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")

    r.raise_for_status()
