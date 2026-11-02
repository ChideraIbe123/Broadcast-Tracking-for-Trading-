"""ReID embedder backed by boxmot's encoder zoo (OSNet, CLIP-ReID, ...).

Default model: osnet-x0-25-msmt17 (small, downloads automatically). A local
torchreid-style checkpoint such as Deep-EIoU's sports OSNet can be passed as a
path. Output is (N, D) L2-normalised numpy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hoops.device import pick_device


class BoxmotEmbedder:
    def __init__(self, model: str | Path = "osnet-x0-25-msmt17", device: str | None = None, precision: str = "fp32"):
        from boxmot.reid import create_reid_encoder

        self.device = pick_device(device)
        self.encoder = create_reid_encoder(str(model), device=self.device, precision=precision)
        self.dim = int(getattr(self.encoder, "embedding_dim", 0) or 0)
        self._n = 0

    def __call__(self, image_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        from boxmot.structures import Boxes, Detections, Frame

        xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        if len(xyxy) == 0:
            return np.zeros((0, self.dim or 1), np.float32)
        self._n += 1
        # boxmot wants an RGB, CHW, uint8 CPU tensor and matching sample ids
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        sid = f"f{self._n}"
        frame = Frame(image=chw, sample_id=sid)
        det = Detections(
            geometry=Boxes(torch.from_numpy(xyxy)),
            scores=torch.ones(len(xyxy)),
            class_ids=torch.zeros(len(xyxy), dtype=torch.long),
            sample_id=sid,
        )
        feats = self.encoder.encode([frame], [det])[0]
        f = feats.detach().cpu().float().numpy()
        if not self.dim:
            self.dim = f.shape[1]
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
