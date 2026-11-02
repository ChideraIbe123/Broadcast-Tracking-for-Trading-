"""Tracker interface and the per-frame track record every tracker emits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hoops.detect.base import Detections


@dataclass
class Track:
    id: int
    xyxy: np.ndarray  # (4,) x1,y1,x2,y2 in pixels
    conf: float
    cls: int
    state: str  # tentative | tracked | lost
    age: int = 0  # frames since last matched detection
    hits: int = 0  # matched detections so far
    embedding: np.ndarray | None = None

    @property
    def foot(self) -> np.ndarray:
        return np.array([(self.xyxy[0] + self.xyxy[2]) / 2, self.xyxy[3]])


class Tracker(Protocol):
    name: str

    def reset(self) -> None: ...

    def update(self, detections: Detections, frame: np.ndarray | None = None) -> list[Track]:
        """Advance one frame. Returns the tracks that are visible (tracked) this frame."""
        ...


def tracks_to_rows(frame_idx: int, segment: int, tracks: list[Track]) -> list[dict]:
    """Rows for the per-video Parquet: one per visible track per frame."""
    return [
        {
            "frame": frame_idx,
            "segment": segment,
            "track_id": t.id,
            "cls": t.cls,
            "x1": float(t.xyxy[0]),
            "y1": float(t.xyxy[1]),
            "x2": float(t.xyxy[2]),
            "y2": float(t.xyxy[3]),
            "conf": float(t.conf),
            "state": t.state,
        }
        for t in tracks
    ]
