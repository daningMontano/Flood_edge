# src/utils/metrics_logger.py
import os
import csv
from pathlib import Path
from datetime import datetime, timezone

import psutil


class MetricsLogger:
    """
    Registra métricas por evento de inferencia en un CSV (append + flush).
    Métricas:
      - CPU total (%)
      - CPU del proceso (%)
      - RAM total (%)
      - RAM usada total (MB)
      - RAM del proceso RSS (MB)
      - tiempo de inferencia (ms) -> se pasa desde el caller
    """
    FIELDS = [
        "timestamp_utc",
        "source",
        "frame_idx",
        "cpu_total_pct",
        "cpu_proc_pct",
        "ram_total_pct",
        "ram_used_mb",
        "proc_rss_mb",
        "inference_ms",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.proc = psutil.Process(os.getpid())

        # Warm-up de contadores (psutil usa deltas entre llamadas).
        psutil.cpu_percent(interval=None)
        self.proc.cpu_percent(interval=None)

        is_new = not self.csv_path.exists()
        self._fh = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)

        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def _sample_cpu_ram(self) -> dict:
        vm = psutil.virtual_memory()

        cpu_total = psutil.cpu_percent(interval=None)
        cpu_proc = self.proc.cpu_percent(interval=None)

        ram_total_pct = float(vm.percent)
        ram_used_mb = float(vm.used) / (1024.0 ** 2)

        rss_mb = float(self.proc.memory_info().rss) / (1024.0 ** 2)

        return {
            "cpu_total_pct": float(cpu_total),
            "cpu_proc_pct": float(cpu_proc),
            "ram_total_pct": ram_total_pct,
            "ram_used_mb": ram_used_mb,
            "proc_rss_mb": rss_mb,
        }

    def log(self, *, inference_ms: float, frame_idx: int = -1, source: str = "unknown") -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "source": source,
            "frame_idx": int(frame_idx),
            "inference_ms": float(inference_ms),
        }
        row.update(self._sample_cpu_ram())

        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
