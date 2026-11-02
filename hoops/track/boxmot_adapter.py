"""Adapter for boxmot trackers (AGPL-3.0): BoT-SORT, OccluBoost, BoostTrack,
ByteTrack, OC-SORT, Deep OC-SORT, StrongSORT, HybridSort.

Input to boxmot: (N, 6) [x1, y1, x2, y2, conf, cls] + BGR frame.
Output rows: [x1, y1, x2, y2, id, score, cls, det_idx].
"""
from __future__ import annotations

import numpy as np

from hoops.detect.base import Detections
from hoops.device import pick_device

from .base import Track

_CLASSES = {
    "botsort": "BotSort",
    "occluboost": "OccluBoost",
    "boosttrack": "BoostTrack",
    "bytetrack": "ByteTrack",
    "ocsort": "OcSort",
    "deepocsort": "DeepOcSort",
    "strongsort": "StrongSort",
    "hybridsort": "HybridSort",
}


class BoxmotAdapter:
    def __init__(self, kind: str = "botsort", reid: str | None = None, device: str | None = None, **kw):
        import boxmot

        self.kind = kind
        self.name = f"boxmot:{kind}" + (f"+{reid}" if reid else "")
        self._cls = getattr(boxmot, _CLASSES[kind])
        self._device = pick_device(device)
        self._reid = reid
        self._kw = kw
        self.reset()

    def reset(self) -> None:
        kw = dict(self._kw)
        if self._reid:
            from boxmot import ReIDConfig

            kw["reid"] = ReIDConfig(self._reid, device=self._device)
        self._t = self._cls(**kw)

    def update(self, detections: Detections, frame: np.ndarray | None = None) -> list[Track]:
        if frame is None:
            raise ValueError("boxmot trackers need the frame")
        d = np.hstack([detections.xyxy, detections.conf[:, None], detections.cls[:, None]]).astype(np.float32)
        out = self._t.update(d, frame)
        if out is None:
            return []
        rows = out if isinstance(out, np.ndarray) else out.to_aabb_rows().cpu().numpy()
        if len(rows) == 0:
            return []
        return [
            Track(int(r[4]), np.asarray(r[:4], float), float(r[5]), int(r[6]), "tracked") for r in rows if r[4] >= 0
        ]
