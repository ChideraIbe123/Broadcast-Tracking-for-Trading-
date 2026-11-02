"""RF-DETR detector wrapper (Apache 2.0).

Default: COCO-pretrained RF-DETR Medium, selecting "person" and "sports ball"
(renamed to "player" / "ball" so downstream code uses one vocabulary). After
Phase 2 fine-tuning, pass `weights=` and the label map becomes the checkpoint's
own class list.

Label-map detail: COCO checkpoints emit COCO category ids (18 -> "dog"), while
fine-tuned checkpoints emit indices into `model.class_names`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hoops.device import pick_device

from .base import Detections

_SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}
COCO_ALIASES = {"person": "player", "sports ball": "ball"}


def _coco_names() -> dict[int, str]:
    from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES

    if isinstance(COCO_CLASSES, dict):
        return {int(k): str(v) for k, v in COCO_CLASSES.items()}
    return {int(k): str(v) for k, v in zip(COCO_CLASSES, COCO_CLASS_NAMES)}


class RFDETRDetector:
    name = "rfdetr"

    def __init__(
        self,
        size: str = "medium",
        weights: str | Path | None = None,
        device: str | None = None,
        conf: float = 0.3,
        resolution: int | None = None,
        keep: tuple[str, ...] | None = ("player", "ball"),
    ):
        import rfdetr

        self.device = pick_device(device)
        cls = getattr(rfdetr, _SIZES[size])
        kw = {"device": self.device}
        if weights:
            kw["pretrain_weights"] = str(weights)
        if resolution:
            kw["resolution"] = resolution
        self.model = cls(**kw)
        self.conf = conf
        if weights:
            names = {i: str(n) for i, n in enumerate(self.model.class_names)}
        else:
            names = {k: COCO_ALIASES.get(v, v) for k, v in _coco_names().items()}
        self.names = names
        self.keep = set(keep) if keep else None
        self.size = size

    def __call__(self, image_bgr: np.ndarray) -> Detections:
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        det = self.model.predict(rgb, threshold=self.conf)
        if det is None or len(det) == 0:
            return Detections.empty(self.names)
        out = Detections(
            det.xyxy.astype(np.float32),
            det.confidence.astype(np.float32),
            det.class_id.astype(int),
            self.names,
        )
        if self.keep:
            out = out.by_name(*self.keep)
        return out

    def optimize(self) -> None:
        """Fuse/compile for faster inference on CUDA; a no-op elsewhere."""
        if self.device.startswith("cuda"):
            self.model.optimize_for_inference()
