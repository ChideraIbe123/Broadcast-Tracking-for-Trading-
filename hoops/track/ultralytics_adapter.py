"""Adapter for Ultralytics trackers (AGPL-3.0): TrackTrack (CVPR 2025, the
current Ultralytics default), BoT-SORT, ByteTrack, OC-SORT, Deep OC-SORT, FastTracker.

Ultralytics trackers take a NumPy-backed "results" object exposing conf, cls,
xywh and boolean indexing; we provide that from our Detections. ReID is enabled
with with_reid=True and model='<path or yolo26n-reid.onnx>' (auto-downloaded).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hoops.detect.base import Detections

from .base import Track


class _Results:
    """Minimal duck type of ultralytics Boxes after .cpu().numpy()."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32)
        self.cls = np.asarray(cls, np.float32)

    @property
    def xywh(self) -> np.ndarray:
        b = self.xyxy
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, m):
        return _Results(self.xyxy[m], self.conf[m], self.cls[m])


class UltralyticsAdapter:
    def __init__(
        self,
        kind: str = "tracktrack",
        with_reid: bool = False,
        reid_model: str = "auto",
        device: str | None = None,
        **overrides,
    ):
        from ultralytics.trackers.track import TRACKER_MAP
        from ultralytics.utils import YAML, IterableSimpleNamespace
        from ultralytics.utils.checks import check_yaml

        from hoops.device import pick_device

        self.kind = kind
        self.name = f"ultralytics:{kind}" + ("+reid" if with_reid else "")
        cfg = YAML.load(check_yaml(f"{kind}.yaml"))
        # The shipped bytetrack/botsort YAMLs assume YOLO confidences (high 0.25, new 0.25);
        # with RF-DETR that promotes 0.3-0.4 crowd detections into tracks. Align with the
        # thresholds every other candidate uses unless the caller overrides them.
        if kind in ("bytetrack", "botsort"):
            cfg.setdefault("track_high_thresh", 0.5)
            cfg["track_high_thresh"] = max(cfg["track_high_thresh"], 0.5)
            cfg["new_track_thresh"] = max(cfg.get("new_track_thresh", 0.0), 0.6)
        cfg.update(overrides)
        if "with_reid" in cfg:
            cfg["with_reid"] = with_reid
            # 'auto' needs the YOLO predictor's native features; standalone we use the ONNX encoder
            cfg["model"] = reid_model if reid_model != "auto" else "yolo26n-reid.onnx"
        cfg["device"] = pick_device(device)
        self._args = IterableSimpleNamespace(**cfg)
        self._cls = TRACKER_MAP[kind]
        self.reset()

    def reset(self) -> None:
        self._t = self._cls(args=self._args)

    def update(self, detections: Detections, frame: np.ndarray | None = None) -> list[Track]:
        res = _Results(detections.xyxy, detections.conf, detections.cls)
        out = self._t.update(res, frame)
        if out is None or len(out) == 0:
            return []
        return [Track(int(r[4]), np.asarray(r[:4], float), float(r[5]), int(r[6]), "tracked") for r in out]
