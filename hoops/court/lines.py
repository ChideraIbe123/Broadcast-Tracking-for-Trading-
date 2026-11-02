"""Line support: does a homography put the court drawing on the painted lines?

A homography fitted to hallucinated landmarks (the "ghost court" a keypoint model
emits for the end it cannot see) is internally consistent and passes every
geometric check. What it cannot do is align with the paint. We project the court
line model through a candidate H into the image and measure the edge strength
under it, compared with the same drawing nudged sideways. Real calibrations score
high; ghosts and replay/close-up frames score near zero.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import court_lines
from .homography import HomographySolution, court_to_image

_SAMPLED: np.ndarray | None = None


def _court_samples(step_m: float = 0.25) -> np.ndarray:
    """Dense (N, 2) court points along every painted line (cached)."""
    global _SAMPLED
    if _SAMPLED is None:
        pts = []
        for poly in court_lines().values():
            for a, b in zip(poly[:-1], poly[1:]):
                n = max(int(np.linalg.norm(b - a) / step_m), 1)
                t = np.linspace(0, 1, n, endpoint=False)[:, None]
                pts.append(a + t * (b - a))
        _SAMPLED = np.vstack(pts)
    return _SAMPLED


def edge_map(image_bgr: np.ndarray, ridge_sigma: float = 2.5) -> np.ndarray:
    """Edge-energy map in [0, 1]. The gradient magnitude is blurred so that the centre
    of a painted line (where the gradient itself is zero, between its two edges) carries
    a ridge response; polarity-agnostic, so white lines on wood and dark lines on paint
    both count. Normalised by the 97th percentile so lighting does not matter."""
    g = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), ridge_sigma)
    hi = np.percentile(m, 97)
    return np.clip(m / max(hi, 1e-6), 0, 1)


def line_support(H: np.ndarray, edges: np.ndarray, shift_px: float = 12.0) -> float:
    """Excess edge energy under the projected court lines over copies shifted sideways.

    ~0 when the drawing is not on the paint; a sharp fit on a clean court scores ~0.3+.
    """
    h, w = edges.shape
    px = court_to_image(H, _court_samples())
    ok = np.isfinite(px).all(axis=1) & (px[:, 0] >= 1) & (px[:, 0] < w - 1) & (px[:, 1] >= 1) & (px[:, 1] < h - 1)
    if ok.sum() < 50:
        return 0.0
    p = px[ok]
    # a near-degenerate H collapses many court lines onto the same pixels (a "fan"); count
    # each 4x4 cell once so piling samples onto one strong edge cannot inflate the score
    cells = np.unique((p / 4).round().astype(int), axis=0)
    if len(cells) < 40:
        return 0.0
    p = cells * 4.0

    def mean_edge(q: np.ndarray) -> float:
        xi = np.clip(q[:, 0].round().astype(int), 0, w - 1)
        yi = np.clip(q[:, 1].round().astype(int), 0, h - 1)
        return float(edges[yi, xi].mean())

    on = mean_edge(p)
    # shifted controls in four directions: the paint is thin, a 7 px shift leaves it
    off = np.mean([mean_edge(p + d) for d in ((shift_px, 0), (-shift_px, 0), (0, shift_px), (0, -shift_px))])
    return float(on - off)


def refine_by_line_support(
    H: np.ndarray,
    image_bgr: np.ndarray,
    max_iter: int = 150,
    edges_coarse: np.ndarray | None = None,
    edges_fine: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Nudge H so the projected court lines sit on the paint (coarse-to-fine).

    Optimises the four PTZ parameters (pan, tilt, roll, log f) of the camera decomposed
    from H, first on a heavily blurred edge map (wide basin), then on the sharp one.
    League-agnostic: it only needs the active court's line model, so it also corrects a
    landmark model trained on another league's floor. Returns (H_refined, support);
    falls back to the input when the decomposition fails or support does not improve.
    """
    from scipy.optimize import minimize

    from .camera import CameraParams, PTZCamera, decompose_homography

    h, w = image_bgr.shape[:2]
    e_fine = edge_map(image_bgr) if edges_fine is None else edges_fine
    s0 = line_support(H, e_fine)
    dec = decompose_homography(np.linalg.inv(H), w / 2, h / 2)
    if dec is None or dec.C[2] <= 0.5:
        return H, s0
    cam = PTZCamera(dec.C, w / 2, h / 2)
    try:
        p, _ = cam.fit(H, dec.params)
    except Exception:
        return H, s0
    x = p.as_array()
    e_coarse = edge_map(image_bgr, ridge_sigma=6.0) if edges_coarse is None else edges_coarse

    def objective(v: np.ndarray, edges: np.ndarray, shift: float) -> float:
        try:
            return -line_support(cam.H_image_to_court(CameraParams.from_array(v)), edges, shift)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 1.0  # worst possible support: steer the simplex away

    for edges, shift, iters in ((e_coarse, 20.0, max_iter), (e_fine, 12.0, max_iter)):
        res = minimize(
            objective,
            x,
            args=(edges, shift),
            method="Nelder-Mead",
            options=dict(initial_simplex=np.vstack([x, x + np.diag([0.01, 0.01, 0.004, 0.02])]), xatol=1e-5, fatol=1e-5, maxiter=iters, maxfev=4 * iters),
        )
        if np.isfinite(res.fun) and res.fun < 1.0:
            x = res.x
    try:
        H_ref = cam.H_image_to_court(CameraParams.from_array(x))
    except np.linalg.LinAlgError:
        return H, s0
    s1 = line_support(H_ref, e_fine)
    return (H_ref, s1) if s1 > s0 else (H, s0)


def select_by_line_support(
    candidates: list[HomographySolution], image_bgr: np.ndarray, min_support: float = 0.02
) -> tuple[HomographySolution | None, list[float]]:
    """Pick the candidate whose court drawing sits on the painted lines.

    Threshold calibrated on the bcd2 test split (2026-10-28): true-label homographies
    score 0.11-0.31, the model's real-end fits ~0.075, ghost courts ~0.004; 0.02 keeps
    92% of real fits and 3% of ghosts."""
    cands = [c for c in candidates if c.ok]
    if not cands:
        return None, []
    edges = edge_map(image_bgr)
    scores = [line_support(c.H, edges) for c in cands]
    best = int(np.argmax(scores))
    if scores[best] < min_support:
        cands[best].reason = f"court lines not on the paint (support {scores[best]:.3f})"
        return None, scores
    return cands[best], scores
