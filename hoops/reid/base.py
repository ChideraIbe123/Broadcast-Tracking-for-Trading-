"""Appearance embedder interface (ReID).

An embedder turns each player crop into a vector such that the same player on
different frames is close and different players are far. Used by the tracker to
re-link identities through occlusions and by offline tracklet merging.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def __call__(self, image_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """(N, dim) L2-normalised embeddings for the given boxes."""
        ...


def crop_boxes(image_bgr: np.ndarray, xyxy: np.ndarray, pad: float = 0.0) -> list[np.ndarray]:
    h, w = image_bgr.shape[:2]
    crops = []
    for x1, y1, x2, y2 in np.asarray(xyxy):
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = int(max(x1 - pad * bw, 0)), int(min(x2 + pad * bw, w))
        y1, y2 = int(max(y1 - pad * bh, 0)), int(min(y2 + pad * bh, h))
        if x2 <= x1 + 1 or y2 <= y1 + 1:
            crops.append(np.zeros((8, 4, 3), np.uint8))
        else:
            crops.append(image_bgr[y1:y2, x1:x2])
    return crops
