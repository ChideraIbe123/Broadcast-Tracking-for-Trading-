"""Detection container shared by every detector and tracker.

Boxes are (N, 4) float32 in x1, y1, x2, y2 pixels. Class ids follow the
detector's own label map; `names` carries the id -> name mapping so downstream
code can select by name ("player", "ball", "referee", "person", ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Detections:
    xyxy: np.ndarray  # (N, 4)
    conf: np.ndarray  # (N,)
    cls: np.ndarray  # (N,) int
    names: dict[int, str] = field(default_factory=dict)
    embeddings: np.ndarray | None = None  # (N, D) optional appearance features

    def __len__(self) -> int:
        return len(self.xyxy)

    @classmethod
    def empty(cls, names: dict[int, str] | None = None) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int), names or {})

    def select(self, mask: np.ndarray) -> Detections:
        emb = None if self.embeddings is None else self.embeddings[mask]
        return Detections(self.xyxy[mask], self.conf[mask], self.cls[mask], self.names, emb)

    def by_name(self, *wanted: str) -> Detections:
        ids = {i for i, n in self.names.items() if n in wanted}
        return self.select(np.isin(self.cls, list(ids)))

    @property
    def xywh(self) -> np.ndarray:
        """Centre x, centre y, width, height (the tracker's measurement space)."""
        b = self.xyxy
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)

    def to_supervision(self):
        import supervision as sv

        return sv.Detections(xyxy=self.xyxy, confidence=self.conf, class_id=self.cls)


class Detector(Protocol):
    names: dict[int, str]

    def __call__(self, image_bgr: np.ndarray) -> Detections: ...


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
