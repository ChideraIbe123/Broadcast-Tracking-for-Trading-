"""Per-frame homography between image pixels and court metres.

Given detected landmarks (pixel positions + confidences) and their known court
positions, solve the 3x3 homography robustly with RANSAC and report residuals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import LANDMARK_XY


@dataclass
class HomographySolution:
    H: np.ndarray | None  # image -> court (metres); None when unsolvable
    inlier_mask: np.ndarray  # bool per supplied landmark
    residuals_px: np.ndarray  # reprojection error in pixels per supplied landmark
    n_input: int
    spread: float  # convex-hull area of inlier pixels / image area
    reason: str = ""
    used_ids: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.H is not None

    @property
    def n_inliers(self) -> int:
        return int(self.inlier_mask.sum())

    @property
    def mean_inlier_error_px(self) -> float:
        if self.n_inliers == 0:
            return float("inf")
        return float(self.residuals_px[self.inlier_mask].mean())


def project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply H to (N, 2) points."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = hom[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return hom[:, :2] / w


def image_to_court(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    return project(H, px)


def court_to_image(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return project(np.linalg.inv(H), xy)


def handedness(H: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Sign of the Jacobian determinant of image->court near the bottom-centre of
    the frame (always floor). Image y points down and court y points up, so a real
    camera above the floor, on either sideline, gives a NEGATIVE value. A positive
    value means the court is mirrored: the classic left/right landmark confusion,
    which no reprojection test can detect because a mirror is still a homography.
    """
    h, w = image_shape[:2]
    p = np.array([[w / 2, 0.9 * h]])
    d = 1.0
    x0 = project(H, p)[0]
    du = project(H, p + [d, 0])[0] - x0
    dv = project(H, p + [0, d])[0] - x0
    return float(du[0] * dv[1] - du[1] * dv[0])


def hull_area_fraction(px: np.ndarray, image_shape: tuple[int, int]) -> float:
    if len(px) < 3:
        return 0.0
    hull = cv2.convexHull(px.astype(np.float32))
    return float(cv2.contourArea(hull) / (image_shape[0] * image_shape[1]))


def solve_homography(
    img_pts: np.ndarray,
    landmark_idx: np.ndarray,
    conf: np.ndarray | None,
    image_shape: tuple[int, int],
    conf_threshold: float = 0.3,
    reproj_px: float = 8.0,
    min_points: int = 4,
    min_spread: float = 0.02,
    court_xy: np.ndarray | None = None,
    require_orientation: bool = True,
    max_retries: int = 2,
) -> HomographySolution:
    """RANSAC homography from detected landmarks.

    img_pts: (N, 2) pixel positions; landmark_idx: (N,) indices into LANDMARKS
    (or rows of `court_xy` when given); conf: (N,) confidences or None.

    With `require_orientation`, a mirrored solution (see `handedness`) is not
    accepted: its inliers are the mirrored landmark identities, so they are dropped
    and RANSAC is re-run on the remaining points, up to `max_retries` times.
    """
    img_pts = np.asarray(img_pts, dtype=float).reshape(-1, 2)
    landmark_idx = np.asarray(landmark_idx, dtype=int).reshape(-1)
    n_all = len(img_pts)
    conf = np.ones(n_all) if conf is None else np.asarray(conf, dtype=float).reshape(-1)
    table = LANDMARK_XY if court_xy is None else np.asarray(court_xy, dtype=float)

    cands = solve_homography_candidates(
        img_pts, landmark_idx, conf, image_shape, conf_threshold, reproj_px, min_points, min_spread, court_xy,
        require_orientation, n_candidates=1, max_retries=max_retries,
    )
    return cands[0]


def solve_homography_candidates(
    img_pts: np.ndarray,
    landmark_idx: np.ndarray,
    conf: np.ndarray | None,
    image_shape: tuple[int, int],
    conf_threshold: float = 0.3,
    reproj_px: float = 8.0,
    min_points: int = 4,
    min_spread: float = 0.02,
    court_xy: np.ndarray | None = None,
    require_orientation: bool = True,
    n_candidates: int = 3,
    max_retries: int = 2,
) -> list[HomographySolution]:
    """Successive RANSAC solutions: after each valid fit, its inliers are removed and
    RANSAC runs again, so a hallucinated "ghost court" and the real one both come
    back as candidates for a later image-based choice (see lines.select_by_line_support).
    Mirrored fits are skipped (their inliers dropped) and never returned. The list
    always has at least one entry; entry 0 is the first valid fit (or the failure)."""
    img_pts = np.asarray(img_pts, dtype=float).reshape(-1, 2)
    landmark_idx = np.asarray(landmark_idx, dtype=int).reshape(-1)
    n_all = len(img_pts)
    conf = np.ones(n_all) if conf is None else np.asarray(conf, dtype=float).reshape(-1)
    table = LANDMARK_XY if court_xy is None else np.asarray(court_xy, dtype=float)

    keep = conf >= conf_threshold
    out: list[HomographySolution] = []
    mirrored = 0
    last = None
    while len(out) < n_candidates and mirrored <= max_retries:
        sol = _solve_once(img_pts, landmark_idx, keep, table, image_shape, reproj_px, min_points, min_spread)
        last = sol
        if not sol.ok:
            break
        keep = keep & ~sol.inlier_mask
        if require_orientation and handedness(sol.H, image_shape) >= 0:
            mirrored += 1
            continue
        out.append(sol)
    if not out:
        if last is None or last.ok:
            last = HomographySolution(None, np.zeros(n_all, bool), np.full(n_all, np.inf), n_all, 0.0)
        if mirrored:
            last.reason = "only mirrored solutions found"
        out.append(last)
    return out


def _solve_once(img_pts, landmark_idx, keep, table, image_shape, reproj_px, min_points, min_spread) -> HomographySolution:
    n_all = len(img_pts)
    empty = HomographySolution(None, np.zeros(n_all, bool), np.full(n_all, np.inf), n_all, 0.0)
    # landmarks the active court does not have (NaN coordinates) cannot be used
    keep = keep & np.isfinite(table[landmark_idx]).all(axis=1)
    if keep.sum() < min_points:
        empty.reason = f"only {int(keep.sum())} usable landmarks"
        return empty

    src = img_pts[keep]
    dst = table[landmark_idx[keep]]
    spread = hull_area_fraction(src, image_shape)
    if spread < min_spread:
        empty.spread = spread
        empty.reason = f"landmarks too clustered/collinear (spread {spread:.3f})"
        return empty

    # Solve in pixel space (dst -> src) so the RANSAC threshold is in pixels.
    H_c2i, mask = cv2.findHomography(
        dst.astype(np.float64), src.astype(np.float64), cv2.RANSAC, reproj_px, maxIters=2000, confidence=0.995
    )
    if H_c2i is None or not np.isfinite(H_c2i).all() or abs(np.linalg.det(H_c2i)) < 1e-12:
        empty.spread = spread
        empty.reason = "findHomography failed"
        return empty

    H = np.linalg.inv(H_c2i)
    H /= H[2, 2]
    # residuals for every supplied landmark (not just the kept ones)
    reproj = project(H_c2i, table[landmark_idx])
    residuals = np.linalg.norm(reproj - img_pts, axis=1)
    inlier = np.zeros(n_all, bool)
    inlier[np.flatnonzero(keep)] = mask.ravel().astype(bool)
    if inlier.sum() < min_points:
        empty.spread = spread
        empty.residuals_px = residuals
        empty.reason = f"only {int(inlier.sum())} RANSAC inliers"
        return empty
    return HomographySolution(
        H=H,
        inlier_mask=inlier,
        residuals_px=residuals,
        n_input=n_all,
        spread=hull_area_fraction(img_pts[inlier], image_shape),
        used_ids=landmark_idx[inlier].tolist(),
    )
