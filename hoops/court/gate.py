"""Calibration quality gate: decide per frame whether the homography is trustworthy.

A wrong homography does not look wrong; it yields plausible positions that are
metres off. The gate turns that silent failure into an explicit label:
    accepted  -> court coordinates are emitted
    uncertain -> emitted but flagged (features may down-weight)
    rejected  -> no court coordinates for this frame (replay, close-up, cut, bad fit)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import cur
from .homography import HomographySolution, court_to_image, handedness, project


@dataclass
class GateConfig:
    min_inliers_accept: int = 6
    min_inliers_uncertain: int = 4
    max_mean_err_px: float = 6.0
    max_mean_err_px_uncertain: float = 12.0
    min_spread: float = 0.03
    max_cond: float = 1e7
    # projected court corners must fall within this many image widths/heights of the frame
    corner_margin: float = 3.0
    # per-frame camera motion bound: max movement of projected court centre in image fractions
    max_centre_shift: float = 0.25
    # agreement between this frame's H and a propagated/previous H (px at four court corners)
    max_disagreement_px: float = 40.0
    # when a line-support score is supplied it replaces the landmark inlier/residual
    # criteria: lines on the paint are direct evidence, landmarks can be biased
    # (e.g. a model trained on another league's floor). Calibrated on 720p broadcast.
    accept_line_support: float = 0.04
    min_line_support: float = 0.02
    # physical plausibility of the camera decomposed from H (broadcast cameras)
    focal_range_frac: tuple[float, float] = (0.4, 25.0)  # focal length / image width
    camera_height_m: tuple[float, float] = (2.5, 45.0)
    camera_lateral_m: float = 80.0  # |x| and |y| of the camera position


@dataclass
class GateResult:
    label: str  # accepted | uncertain | rejected
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.label == "accepted"

    @property
    def usable(self) -> bool:
        return self.label in ("accepted", "uncertain")


def _court_corners() -> np.ndarray:
    c = cur()
    return np.array([[-c.half_l, -c.half_w], [c.half_l, -c.half_w], [c.half_l, c.half_w], [-c.half_l, c.half_w]])


def _degenerate(H: np.ndarray, max_cond: float) -> str | None:
    if not np.isfinite(H).all():
        return "H has non-finite entries"
    det = np.linalg.det(H)
    if abs(det) < 1e-12:
        return "H is singular"
    cond = np.linalg.cond(H)
    if cond > max_cond:
        return f"H ill-conditioned (cond={cond:.2e})"
    return None




def assess(
    sol: HomographySolution,
    image_shape: tuple[int, int],
    prev_H: np.ndarray | None = None,
    cfg: GateConfig | None = None,
    line_support: float | None = None,
) -> GateResult:
    cfg = cfg or GateConfig()
    h, w = image_shape[:2]
    reasons: list[str] = []
    metrics: dict[str, float] = {}

    if not sol.ok:
        return GateResult("rejected", [sol.reason or "no homography"], metrics)

    H = sol.H
    metrics["n_inliers"] = sol.n_inliers
    metrics["mean_err_px"] = sol.mean_inlier_error_px
    metrics["spread"] = sol.spread

    bad = _degenerate(H, cfg.max_cond)
    if bad:
        return GateResult("rejected", [bad], metrics)

    # projected court corners must land somewhere sane relative to the frame
    corners = _court_corners()
    corners_px = court_to_image(H, corners)
    if not np.isfinite(corners_px).all():
        return GateResult("rejected", ["court corners project to infinity"], metrics)
    m = cfg.corner_margin
    if (np.abs(corners_px[:, 0] - w / 2) > m * w).any() or (np.abs(corners_px[:, 1] - h / 2) > m * h).any():
        return GateResult("rejected", ["court corners far outside the frame"], metrics)

    # the court must appear with the camera above it: image-y of the far sideline < near sideline
    # (i.e. the projected court is not upside down). We check the mean of the top vs bottom edge.
    top = corners_px[2:4, 1].mean()
    bottom = corners_px[0:2, 1].mean()
    metrics["court_height_px"] = float(abs(bottom - top))
    if metrics["court_height_px"] < 0.05 * h:
        return GateResult("rejected", ["court projects to a sliver (extreme angle or bad fit)"], metrics)

    # absolute handedness: a mirrored court would put the camera below the floor
    metrics["handedness"] = handedness(H, image_shape)
    if metrics["handedness"] >= 0:
        return GateResult("rejected", ["court mirrored (left/right landmark confusion)"], metrics)

    # physical camera: focal length, height and distance must be broadcast-like. A
    # near-degenerate H (court drawn as a fan of lines) decomposes into nonsense here.
    from .camera import decompose_homography  # local import: camera.py imports this module

    dec = decompose_homography(np.linalg.inv(H), w / 2, h / 2)
    if dec is None:
        return GateResult("rejected", ["camera decomposition failed"], metrics)
    metrics["focal_px"], metrics["cam_z"], metrics["cam_y"] = float(dec.f), float(dec.C[2]), float(dec.C[1])
    lo, hi = cfg.focal_range_frac
    if not (lo * w <= dec.f <= hi * w):
        return GateResult("rejected", [f"implausible focal length {dec.f:.0f}px"], metrics)
    if not (cfg.camera_height_m[0] <= dec.C[2] <= cfg.camera_height_m[1]):
        return GateResult("rejected", [f"implausible camera height {dec.C[2]:.1f}m"], metrics)
    if abs(dec.C[0]) > cfg.camera_lateral_m or abs(dec.C[1]) > cfg.camera_lateral_m:
        return GateResult("rejected", [f"implausible camera position ({dec.C[0]:.0f}, {dec.C[1]:.0f})m"], metrics)

    if prev_H is not None:
        centre_prev = court_to_image(prev_H, np.zeros((1, 2)))[0]
        centre_now = court_to_image(H, np.zeros((1, 2)))[0]
        shift = np.linalg.norm((centre_now - centre_prev) / np.array([w, h]))
        metrics["centre_shift"] = float(shift)
        if shift > cfg.max_centre_shift:
            reasons.append(f"camera jump {shift:.2f} of frame")
        disagreement = np.linalg.norm(court_to_image(prev_H, corners) - corners_px, axis=1).mean()
        metrics["disagreement_px"] = float(disagreement)
        if disagreement > cfg.max_disagreement_px:
            reasons.append(f"disagrees with previous H by {disagreement:.0f}px")

    # accept / uncertain / reject on fit quality
    if line_support is not None:
        metrics["line_support"] = float(line_support)
        if line_support >= cfg.accept_line_support and not reasons:
            return GateResult("accepted", [], metrics)
        if line_support >= cfg.min_line_support:
            return GateResult("uncertain", reasons + [f"line support {line_support:.3f}"], metrics)
        return GateResult("rejected", reasons + [f"court lines not on the paint ({line_support:.3f})"], metrics)
    n, e, s = sol.n_inliers, sol.mean_inlier_error_px, sol.spread
    if n >= cfg.min_inliers_accept and e <= cfg.max_mean_err_px and s >= cfg.min_spread and not reasons:
        return GateResult("accepted", [], metrics)
    if n >= cfg.min_inliers_uncertain and e <= cfg.max_mean_err_px_uncertain:
        if n < cfg.min_inliers_accept:
            reasons.append(f"only {n} inliers")
        if e > cfg.max_mean_err_px:
            reasons.append(f"mean error {e:.1f}px")
        if s < cfg.min_spread:
            reasons.append(f"low spread {s:.3f}")
        return GateResult("uncertain", reasons, metrics)
    reasons.append(f"{n} inliers, {e:.1f}px mean error")
    return GateResult("rejected", reasons, metrics)


def foot_points_to_court(H: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Bottom-centre of each box -> court metres. boxes: (N, 4) x1,y1,x2,y2."""
    b = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    feet = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
    return project(H, feet)
