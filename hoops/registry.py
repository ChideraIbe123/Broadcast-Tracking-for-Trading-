"""Build detectors / trackers / embedders from short names (used by the CLI and the
benchmark harness so every candidate is one string away)."""
from __future__ import annotations

from hoops.track.ours import HoopsTracker, HoopsTrackerConfig

TRACKER_CHOICES = [
    "hoops:kalman_cv",
    "hoops:kalman_ca",
    "hoops:imm",
    "hoops:particle",
    "trackers:botsort",
    "trackers:bytetrack",
    "trackers:ocsort",
    "trackers:sort",
    "boxmot:botsort",
    "boxmot:occluboost",
    "boxmot:boosttrack",
    "boxmot:bytetrack",
    "ultralytics:tracktrack",
    "ultralytics:botsort",
    "ultralytics:bytetrack",
]


def make_detector(name: str = "rfdetr:medium", device: str | None = None, conf: float = 0.3, weights=None):
    kind, _, variant = name.partition(":")
    if kind == "rfdetr":
        from hoops.detect.rfdetr_detector import RFDETRDetector

        return RFDETRDetector(size=variant or "medium", device=device, conf=conf, weights=weights)
    raise ValueError(f"unknown detector {name}")


def make_embedder(name: str | None, device: str | None = None):
    if not name or name == "none":
        return None
    kind, _, variant = name.partition(":")
    if kind == "boxmot":
        from hoops.reid.boxmot_reid import BoxmotEmbedder

        return BoxmotEmbedder(variant or "osnet-x0-25-msmt17", device=device)
    raise ValueError(f"unknown embedder {name}")


def make_tracker(name: str = "hoops:kalman_cv", embedder=None, device: str | None = None, **kw):
    kind, _, variant = name.partition(":")
    if kind == "hoops":
        cfg = HoopsTrackerConfig(motion=variant or "kalman_cv", **kw)
        return HoopsTracker(cfg, embedder=embedder)
    if kind == "trackers":
        from hoops.track.trackers_adapter import RoboflowTrackerAdapter

        return RoboflowTrackerAdapter(variant or "botsort", **kw)
    if kind == "boxmot":
        from hoops.track.boxmot_adapter import BoxmotAdapter

        return BoxmotAdapter(variant or "botsort", device=device, **kw)
    if kind == "ultralytics":
        from hoops.track.ultralytics_adapter import UltralyticsAdapter

        return UltralyticsAdapter(variant or "tracktrack", **kw)
    raise ValueError(f"unknown tracker {name}")
