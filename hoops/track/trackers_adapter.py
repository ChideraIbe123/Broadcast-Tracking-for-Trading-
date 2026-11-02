"""Adapters for Roboflow's `trackers` library (Apache 2.0): SORT, ByteTrack,
OC-SORT, BoT-SORT (with camera-motion compensation), C-BIoU."""
from __future__ import annotations

import numpy as np

from hoops.detect.base import Detections

from .base import Track

_CLASSES = {
    "sort": "SORTTracker",
    "bytetrack": "ByteTrackTracker",
    "ocsort": "OCSORTTracker",
    "botsort": "BoTSORTTracker",
    "cbiou": "CBIoUTracker",
}


class RoboflowTrackerAdapter:
    def __init__(self, kind: str = "botsort", **kw):
        import trackers

        self.kind = kind
        self.name = f"trackers:{kind}"
        self._cls = getattr(trackers, _CLASSES[kind])
        self._kw = kw
        self.reset()

    def reset(self) -> None:
        self._t = self._cls(**self._kw)

    def update(self, detections: Detections, frame: np.ndarray | None = None) -> list[Track]:
        sv_det = detections.to_supervision()
        try:
            out = self._t.update(sv_det, frame)
        except TypeError:  # trackers without a frame argument (no CMC)
            out = self._t.update(sv_det)
        if out is None or len(out) == 0 or out.tracker_id is None:
            return []
        tracks = []
        for xyxy, conf, cls, tid in zip(out.xyxy, out.confidence, out.class_id, out.tracker_id):
            if tid is None or tid < 0:
                continue
            tracks.append(Track(int(tid), np.asarray(xyxy, float), float(conf), int(cls), "tracked"))
        return tracks
