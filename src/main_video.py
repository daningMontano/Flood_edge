# src/main_video.py
# ------------------------------------------------------------
# Procesa un video local, ejecuta segmentación cada N frames y:
#  - calcula severidad por puntos dentro de la máscara
#  - envía alerta Telegram solo si cambia la severidad (medio/alto)
#  - guarda métricas en CSV: CPU, RAM, tiempo de inferencia
# ------------------------------------------------------------

import os
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
from dotenv import load_dotenv

from src.inference.detector import FloodDetectorEdge
from src.preprocess.image import preprocess_image
from src.postprocess.mask import postprocess_mask, overlay_mask_with_points
from src.io.points_loader import load_points_csv
from src.io.video_loader import iter_video_frames
from src.geometry.mask_points import labeled_points_in_mask
from src.actions.telegram_alert import send_alert


# --- Rutas (puedes sobreescribir con variables de entorno) ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/flood_segmentation_dinov3.onnx")
POINTS_VIDEO_PATH = os.getenv("POINTS_VIDEO_PATH", "data/alert_points/points_video4.csv")
VIDEO_PATH = os.getenv("VIDEO_PATH", "data/samples/video_test3.mp4")

# Procesar 1 de cada N frames (para performance)
VIDEO_EVERY_N = int(os.getenv("VIDEO_EVERY_N", "150"))

# Logs: se guarda en <raíz_proyecto>/logs/
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../<repo>/src/main_video.py -> .../<repo>
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_severity(points_inside) -> str:
    """
    Severidad global:
      - si hay "alto" => alto
      - si hay "medio" => medio
      - caso contrario => bajo
    """
    has_medium = False
    for p in points_inside:
        label = (p.get("label") or p.get("Label") or "").strip().lower()
        if label == "alto":
            return "alto"
        if label == "medio":
            has_medium = True
    return "medio" if has_medium else "bajo"


def _metrics_csv_path() -> Path:
    """
    Genera un CSV con timestamp para evitar sobreescritura.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"metrics_video_{stamp}.csv"


def _open_metrics_writer(csv_path: Path):
    """
    Abre el CSV en append y asegura encabezados.
    """
    header = [
        "timestamp_utc",
        "source",
        "video_path",
        "every_n",
        "frame_idx",
        "severity",
        "points_inside",
        "inference_ms",
        "system_cpu_percent",
        "process_cpu_percent",
        "process_rss_mb",
        "system_ram_percent",
        "alert_sent",
        "alert_error",
    ]

    f = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=header)

    # Escribe header solo si el archivo está vacío
    if f.tell() == 0:
        writer.writeheader()
        f.flush()

    return f, writer


def main():
    # 1) Variables de entorno
    load_dotenv()

    # 2) Modelo + puntos
    detector = FloodDetectorEdge(MODEL_PATH)
    points = load_points_csv(POINTS_VIDEO_PATH)

    # 3) Métricas (psutil)
    proc = psutil.Process(os.getpid())
    # Primer llamado para que cpu_percent() no devuelva 0 siempre
    psutil.cpu_percent(interval=None)
    proc.cpu_percent(interval=None)

    # 4) CSV de métricas
    csv_path = Path(os.getenv("METRICS_CSV_PATH", "")) if os.getenv("METRICS_CSV_PATH") else _metrics_csv_path()
    csv_file, csv_writer = _open_metrics_writer(csv_path)

    last_severity = "bajo"

    try:
        for frame_idx, frame in iter_video_frames(VIDEO_PATH, every_n=VIDEO_EVERY_N):
            # ---- INFERENCIA (medición de tiempo) ----
            input_tensor = preprocess_image(frame)

            t0 = time.perf_counter()
            output = detector.predict(input_tensor)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # ---- Postproceso: máscara binaria en tamaño original ----
            prob_map = output[0, 0]
            mask = postprocess_mask(prob_map, frame.shape[:2])

            # ---- Puntos dentro de la máscara ----
            points_inside = labeled_points_in_mask(mask, points)

            # ---- Overlay (solo RAM) ----
            overlay = overlay_mask_with_points(
                image=frame,
                binary_mask=mask,
                points_all=points,
                points_inside=points_inside,
            )

            # ---- Severidad + lógica de alerta por cambio de estado ----
            current_severity = get_severity(points_inside)

            alert_sent = 0
            alert_error = ""
            if current_severity != last_severity:
                if current_severity in ("medio", "alto"):
                    try:
                        send_alert(overlay, points_inside)
                        alert_sent = 1
                    except Exception as e:
                        # No se imprime; queda registrado en el CSV
                        alert_error = str(e)[:300]
                last_severity = current_severity

            # ---- Métricas CPU/RAM ----
            system_cpu = psutil.cpu_percent(interval=None)
            process_cpu = proc.cpu_percent(interval=None)
            process_rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
            system_ram_percent = psutil.virtual_memory().percent

            # ---- Escribir fila CSV ----
            csv_writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "source": "video",
                    "video_path": VIDEO_PATH,
                    "every_n": VIDEO_EVERY_N,
                    "frame_idx": frame_idx,
                    "severity": current_severity,
                    "points_inside": len(points_inside),
                    "inference_ms": round(inference_ms, 3),
                    "system_cpu_percent": system_cpu,
                    "process_cpu_percent": process_cpu,
                    "process_rss_mb": round(process_rss_mb, 3),
                    "system_ram_percent": system_ram_percent,
                    "alert_sent": alert_sent,
                    "alert_error": alert_error,
                }
            )
            csv_file.flush()

            # ---- Limpieza explícita (opcional, útil en edge) ----
            del overlay, mask, output, input_tensor

    finally:
        try:
            csv_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

