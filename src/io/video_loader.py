# src/io/video_loader.py
import cv2
from typing import Iterator, Tuple
import numpy as np


def iter_video_frames(video_path: str, every_n: int = 3) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Lee un video y entrega (frame_index, frame) solo cada 'every_n' fotogramas.
    frame_index es 0-based.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_n == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
