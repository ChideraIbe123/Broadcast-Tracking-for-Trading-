"""Frame reader for broadcast video (OpenCV backend, ffmpeg under the hood)."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    info = VideoInfo(
        Path(path),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def frames(
    path: str | Path,
    start: int = 0,
    end: int | None = None,
    stride: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR image) for frames in [start, end) every `stride`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    try:
        while True:
            if end is not None and i >= end:
                break
            ok, img = cap.read()
            if not ok:
                break
            if (i - start) % stride == 0:
                yield i, img
            i += 1
    finally:
        cap.release()


class VideoWriter:
    """Thin wrapper to write annotated frames next to the input (mp4v)."""

    def __init__(self, path: str | Path, fps: float, shape: tuple[int, int]):
        h, w = shape
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img: np.ndarray) -> None:
        self._w.write(img)

    def close(self) -> None:
        self._w.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
