"""Landmark sources for the calibrator: where do the per-frame court points come from?

  ModelLandmarks  : a trained keypoint model (Ultralytics YOLO-pose or RF-DETR
                    keypoint preview) whose keypoint order equals LANDMARK_IDS.
  ManualLandmarks : hand-clicked points on a few keyframes (JSON); the camera
                    model carries the calibration between keyframes via floor flow.

Each source returns (img_pts (N,2), landmark_idx (N,), conf (N,)) for a frame.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from hoops.device import pick_device

from .geometry import LANDMARK_IDS, landmark_index


# Roboflow "NBA court keypoint detection" (22 keypoints) -> our landmark ids.
# Their frame: "top" = far sideline (+y here), "right" = +x; "neutral" = the 28 ft
# sideline hash marks, "bottle" = paint corner on the baseline, "ft" = free-throw
# line end, "three" = three-point line / baseline junction, "median" = half-court line ends.
ROBOFLOW22 = [
    "top_median", "bot_median", "top_right_neutral", "top_left_neutral", "bot_left_neutral", "bot_right_neutral",
    "top_right_corner", "top_right_three", "top_right_bottle", "top_right_ft", "bot_right_ft", "bot_right_bottle",
    "bot_right_three", "bot_right_corner", "top_left_corner", "top_left_three", "top_left_bottle", "top_left_ft",
    "bot_left_ft", "bot_left_bottle", "bot_left_three", "bot_left_corner",
]
ROBOFLOW22_TO_HOOPS = {
    "top_median": "T_half_side",
    "bot_median": "B_half_side",
    "top_right_neutral": "RT_hash",
    "top_left_neutral": "LT_hash",
    "bot_left_neutral": "LB_hash",
    "bot_right_neutral": "RB_hash",
    "top_right_corner": "RT_corner",
    "top_right_three": "RT_three_base",
    "top_right_bottle": "RT_paint_base",
    "top_right_ft": "RT_paint_ft",
    "bot_right_ft": "RB_paint_ft",
    "bot_right_bottle": "RB_paint_base",
    "bot_right_three": "RB_three_base",
    "bot_right_corner": "RB_corner",
    "top_left_corner": "LT_corner",
    "top_left_three": "LT_three_base",
    "top_left_bottle": "LT_paint_base",
    "top_left_ft": "LT_paint_ft",
    "bot_left_ft": "LB_paint_ft",
    "bot_left_bottle": "LB_paint_base",
    "bot_left_three": "LB_three_base",
    "bot_left_corner": "LB_corner",
}
# horizontal image flip swaps left<->right: index permutation for YOLO-pose flip_idx
ROBOFLOW22_FLIP_IDX = [
    ROBOFLOW22.index(n.replace("right", "§").replace("left", "right").replace("§", "left")) for n in ROBOFLOW22
]

# Roboflow "basketball-court-detection-2" (roboflow-jvuqo, CC BY 4.0): 33 numbered
# vertices of a left-to-right court diagram. Decoded from rendered frames and verified
# geometrically (1.1 px residual over 1,216 images), see the 2026-10-28 dataset report.
BCD33 = [
    "01", "02", "04", "05", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17",
    "19", "21", "23",
    "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "37", "38", "40", "41",
]
BCD33_TO_HOOPS = {
    "01": "LT_corner", "02": "LT_three_base", "04": "LT_paint_base", "05": "LB_paint_base",
    "07": "LB_three_base", "08": "LB_corner", "09": "L_rim", "10": "LT_three_break",
    "11": "LB_three_break", "12": "LT_paint_ft", "13": "L_ft_mid", "14": "LB_paint_ft",
    "15": "LT_hash", "16": "L_three_apex", "17": "LB_hash",
    "19": "T_half_side", "21": "C_centre", "23": "B_half_side",
    "25": "RT_hash", "26": "R_three_apex", "27": "RB_hash", "28": "RT_paint_ft", "29": "R_ft_mid",
    "30": "RB_paint_ft", "31": "RT_three_break", "32": "RB_three_break", "33": "R_rim",
    "34": "RT_corner", "35": "RT_three_base", "37": "RT_paint_base", "38": "RB_paint_base",
    "40": "RB_three_base", "41": "RB_corner",
}


def _flip_idx_from_mapping(names: list[str], mapping: dict[str, str]) -> list[int]:
    from .geometry import mirror_id

    inv = {v: k for k, v in mapping.items()}
    return [names.index(inv[mirror_id(mapping[n])]) for n in names]


BCD33_FLIP_IDX = _flip_idx_from_mapping(BCD33, BCD33_TO_HOOPS)

SCHEMAS = {
    "roboflow22": (ROBOFLOW22, ROBOFLOW22_TO_HOOPS, ROBOFLOW22_FLIP_IDX),
    "bcd33": (BCD33, BCD33_TO_HOOPS, BCD33_FLIP_IDX),
}


class LandmarkSource(Protocol):
    def __call__(self, frame_idx: int, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def _empty():
    return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)


class ManualLandmarks:
    """JSON: {"<frame_idx>": {"LT_corner": [u, v], "R_three_apex": [u, v], ...}, ...}"""

    def __init__(self, path: str | Path):
        raw = json.loads(Path(path).read_text())
        self.frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for k, pts in raw.items():
            ids = [landmark_index(n) for n in pts if n in LANDMARK_IDS]
            uv = np.array([pts[n] for n in pts if n in LANDMARK_IDS], float).reshape(-1, 2)
            self.frames[int(k)] = (uv, np.array(ids, int), np.ones(len(ids)))

    def __call__(self, frame_idx: int, image_bgr: np.ndarray):
        return self.frames.get(frame_idx, _empty())


class ModelLandmarks:
    """`schema`: "hoops" if the model's keypoint order is LANDMARK_IDS, otherwise one of
    SCHEMAS ("roboflow22", "bcd33") for a model trained on that dataset's keypoint order."""

    def __init__(self, weights: str | Path, kind: str = "yolo", device: str | None = None, conf: float = 0.3, schema: str = "bcd33"):
        self.kind = kind
        self.conf = conf
        self.device = pick_device(device)
        if schema in SCHEMAS:
            names, mapping, _ = SCHEMAS[schema]
            self.kp_to_landmark = np.array([landmark_index(mapping[n]) for n in names])
        else:
            self.kp_to_landmark = np.arange(len(LANDMARK_IDS))
        if kind == "yolo":
            from ultralytics import YOLO

            self.model = YOLO(str(weights))
        elif kind == "rfdetr":
            from rfdetr import RFDETRKeypointPreview

            self.model = RFDETRKeypointPreview(pretrain_weights=str(weights), device=self.device)
        else:
            raise ValueError(kind)

    def __call__(self, frame_idx: int, image_bgr: np.ndarray):
        if self.kind == "yolo":
            r = self.model.predict(image_bgr, verbose=False, device=self.device, conf=0.1)[0]
            if r.keypoints is None or len(r.keypoints) == 0:
                return _empty()
            # the court is one "instance": take the highest-confidence detection
            k = r.keypoints
            best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
            xy = k.xy[best].cpu().numpy()
            c = k.conf[best].cpu().numpy() if k.conf is not None else np.ones(len(xy))
        else:
            out = self.model.predict(np.ascontiguousarray(image_bgr[:, :, ::-1]), threshold=0.1)
            if out is None or len(out) == 0:
                return _empty()
            xy = np.asarray(out.xy[0] if hasattr(out, "xy") else out.keypoints_xy[0])
            c = np.asarray(out.confidence[0]) if hasattr(out, "confidence") else np.ones(len(xy))
        keep = (c >= self.conf) & (xy[:, 0] > 0) & (xy[:, 1] > 0)
        idx = np.flatnonzero(keep)
        return xy[idx], self.kp_to_landmark[idx], c[idx]


def make_landmark_source(spec: str, device: str | None = None, conf: float = 0.3) -> LandmarkSource:
    """manual:points.json | yolo:weights.pt[@schema[:conf]] | rfdetr:weights.pth[@schema[:conf]]"""
    kind, _, arg = spec.partition(":")
    if kind == "manual":
        return ManualLandmarks(arg)
    if kind in ("yolo", "rfdetr"):
        weights, _, rest = arg.partition("@")
        schema, _, conf_s = rest.partition(":")
        return ModelLandmarks(weights, kind=kind, device=device, schema=schema or "bcd33", conf=float(conf_s) if conf_s else conf)
    raise ValueError(f"unknown landmark source {spec}")
