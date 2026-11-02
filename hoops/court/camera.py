"""Temporal camera model for broadcast calibration.

A broadcast camera does not move; it pans, tilts, zooms (and may roll slightly).
So instead of smoothing the eight numbers of a homography, we describe the camera
by four parameters per frame, (pan, tilt, roll, log f), with its position C fixed
per segment, and run a Kalman filter on those. Two kinds of measurement feed it:

  * landmarks  -> homography (RANSAC) -> parameters        (absolute, when visible)
  * floor optical flow between consecutive frames          (relative, almost always)

The filtered parameters give the output homography. Everything is validated by
re-projecting the landmarks through that final H.

Conventions: world frame = court frame (metres, z up). Camera frame x right,
y down, z forward. R maps world -> camera; its rows are the camera axes in world
coordinates. Pan is positive to the right (+x court), tilt positive looking down.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from hoops.motion.kalman import LinearKalman

from .gate import GateConfig, GateResult, assess
from .geometry import cur
from .homography import HomographySolution, court_to_image, project, solve_homography, solve_homography_candidates
from .lines import edge_map, line_support, refine_by_line_support, select_by_line_support

UP = np.array([0.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------
def rotation_from_ptr(pan: float, tilt: float, roll: float) -> np.ndarray:
    fwd = np.array([np.sin(pan) * np.cos(tilt), np.cos(pan) * np.cos(tilt), -np.sin(tilt)])
    right0 = np.cross(fwd, UP)
    right0 /= np.linalg.norm(right0)
    down0 = np.cross(fwd, right0)
    right = np.cos(roll) * right0 + np.sin(roll) * down0
    down = -np.sin(roll) * right0 + np.cos(roll) * down0
    return np.stack([right, down, fwd])


def ptr_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    right, _, fwd = R[0], R[1], R[2]
    pan = float(np.arctan2(fwd[0], fwd[1]))
    tilt = float(np.arctan2(-fwd[2], np.hypot(fwd[0], fwd[1])))
    right0 = np.cross(fwd, UP)
    right0 /= np.linalg.norm(right0)
    down0 = np.cross(fwd, right0)
    roll = float(np.arctan2(right @ down0, right @ right0))
    return pan, tilt, roll


# --------------------------------------------------------------------------
# PTZ camera
# --------------------------------------------------------------------------
@dataclass
class CameraParams:
    pan: float
    tilt: float
    roll: float
    logf: float

    @property
    def f(self) -> float:
        return float(np.exp(self.logf))

    def as_array(self) -> np.ndarray:
        return np.array([self.pan, self.tilt, self.roll, self.logf])

    @classmethod
    def from_array(cls, a: np.ndarray) -> CameraParams:
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))


class PTZCamera:
    """Fixed-position camera; H depends only on (pan, tilt, roll, f)."""

    def __init__(self, C: np.ndarray, cx: float, cy: float):
        self.C = np.asarray(C, dtype=float)
        self.cx, self.cy = float(cx), float(cy)

    def K(self, f: float) -> np.ndarray:
        return np.array([[f, 0, self.cx], [0, f, self.cy], [0, 0, 1.0]])

    def H_court_to_image(self, p: CameraParams) -> np.ndarray:
        R = rotation_from_ptr(p.pan, p.tilt, p.roll)
        P = self.K(p.f) @ np.hstack([R, (-R @ self.C)[:, None]])
        H = P[:, [0, 1, 3]]
        return H / H[2, 2]

    def H_image_to_court(self, p: CameraParams) -> np.ndarray:
        Hc = self.H_court_to_image(p)
        if not np.isfinite(Hc).all() or abs(np.linalg.det(Hc)) < 1e-14:
            raise np.linalg.LinAlgError("degenerate camera parameters")
        H = np.linalg.inv(Hc)
        if abs(H[2, 2]) < 1e-14:
            raise np.linalg.LinAlgError("degenerate homography")
        return H / H[2, 2]

    def fit(self, H_i2c: np.ndarray, init: CameraParams | None = None) -> tuple[CameraParams, float]:
        """Parameters whose homography best reproduces H_i2c on a court grid.
        Returns (params, rms pixel error)."""
        pts = _grid()
        target = court_to_image(H_i2c, pts)
        if init is None:
            dec = decompose_homography(np.linalg.inv(H_i2c), self.cx, self.cy)
            init = dec.params if dec is not None else CameraParams(0.0, 0.3, 0.0, np.log(1500.0))

        def resid(x):
            Hc = self.H_court_to_image(CameraParams.from_array(x))
            return (project(Hc, pts) - target).ravel()

        sol = least_squares(resid, init.as_array(), method="lm", max_nfev=200)
        rms = float(np.sqrt(np.mean(sol.fun**2)))
        return CameraParams.from_array(sol.x), rms


def _grid() -> np.ndarray:
    c = cur()
    xs = np.linspace(-c.half_l, c.half_l, 7)
    ys = np.linspace(-c.half_w, c.half_w, 5)
    return np.array([[x, y] for x in xs for y in ys])


@dataclass
class Decomposition:
    f: float
    R: np.ndarray
    t: np.ndarray
    C: np.ndarray
    params: CameraParams


def decompose_homography(H_c2i: np.ndarray, cx: float, cy: float) -> Decomposition | None:
    """Recover focal length and pose from a court->image homography (square
    pixels, known principal point). Single-view plane calibration: with
    omega = diag(a, a, 1), a = 1/f^2, the two constraints h1'w h2 = 0 and
    h1'w h1 = h2'w h2 give a by least squares."""
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    H = T @ H_c2i
    h1, h2 = H[:, 0], H[:, 1]
    A = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = -np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    a = float(A @ b / max(A @ A, 1e-12))
    if not np.isfinite(a) or a <= 0:
        return None
    f = 1.0 / np.sqrt(a)
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    M = Kinv @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:  # court must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    U, _, Vt = np.linalg.svd(np.stack([r1, r2, r3], axis=1))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    C = -R.T @ t
    pan, tilt, roll = ptr_from_rotation(R)
    return Decomposition(f, R, t, C, CameraParams(pan, tilt, roll, float(np.log(f))))


# --------------------------------------------------------------------------
# floor optical flow (relative motion between frames)
# --------------------------------------------------------------------------
class FloorFlow:
    """Homography of the floor between consecutive frames from sparse optical flow,
    using only points inside the projected court and outside player boxes."""

    def __init__(self, downscale: int = 2, max_corners: int = 600, min_inliers: int = 25):
        self.ds = downscale
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = self.prev_pts = None

    def __call__(self, frame_bgr: np.ndarray, H_prev_i2c: np.ndarray | None, boxes_xyxy: np.ndarray | None) -> np.ndarray | None:
        """Return dH (image prev -> image curr, full-res pixels) or None."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.ds > 1:
            gray = cv2.resize(gray, None, fx=1 / self.ds, fy=1 / self.ds, interpolation=cv2.INTER_AREA)
        dH = None
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= self.min_inliers:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
            ok = st.ravel() == 1
            if ok.sum() >= self.min_inliers:
                M, inl = cv2.findHomography(self.prev_pts[ok], nxt[ok], cv2.RANSAC, 2.0)
                if M is not None and inl is not None and inl.sum() >= self.min_inliers:
                    S = np.diag([self.ds, self.ds, 1.0])
                    dH = S @ M @ np.linalg.inv(S)
        # features for the next step: inside the court polygon, outside boxes
        mask = np.full(gray.shape, 255, np.uint8)
        if H_prev_i2c is not None:
            c = cur()
            corners = court_to_image(H_prev_i2c, np.array([[-c.half_l, -c.half_w], [c.half_l, -c.half_w], [c.half_l, c.half_w], [-c.half_l, c.half_w]]))
            if np.isfinite(corners).all():
                poly = (corners / self.ds).astype(np.int32).reshape(-1, 1, 2)
                mask[:] = 0
                cv2.fillPoly(mask, [poly], 255)
        if boxes_xyxy is not None:
            for x1, y1, x2, y2 in np.asarray(boxes_xyxy) / self.ds:
                mask[max(int(y1), 0) : int(y2) + 1, max(int(x1), 0) : int(x2) + 1] = 0
        self.prev_pts = cv2.goodFeaturesToTrack(gray, self.max_corners, 0.01, 6, mask=mask, blockSize=3)
        self.prev_gray = gray
        return dH


# --------------------------------------------------------------------------
# calibrator
# --------------------------------------------------------------------------
@dataclass
class CalibratorConfig:
    gate: GateConfig = None  # type: ignore[assignment]
    n_init_frames: int = 5  # accepted frames used to fix the camera position C
    q_angle: float = 0.002  # process noise (rad / frame) on pan/tilt/roll position
    q_rate: float = 0.002  # process noise on their rates
    r_landmark: tuple[float, float, float, float] = (0.003, 0.003, 0.003, 0.01)
    r_flow: tuple[float, float, float, float] = (0.006, 0.006, 0.006, 0.02)
    max_stale: int = 15  # frames without any measurement before rejecting
    # a measured H the PTZ model cannot reproduce is discarded. Real labelled broadcast
    # frames fit the rigid PTZ model to ~20-45 px rms (lens/principal-point effects), so
    # this must be loose; the landmark residual check on the filtered H guards accuracy.
    max_fit_rms_px: float = 60.0
    good_fit_rms_px: float = 5.0  # fits worse than this get measurement noise scaled by rms/this
    # choose among successive RANSAC candidates by edge support under the projected
    # court lines (defeats hallucinated "ghost courts"); needs the frame
    use_line_support: bool = True
    min_line_support: float = 0.02
    # after choosing a candidate (or propagating by flow), nudge H onto the painted lines
    refine_lines: bool = True
    refine_iters: int = 120

    def __post_init__(self):
        if self.gate is None:
            self.gate = GateConfig()


@dataclass
class CalibrationFrame:
    H: np.ndarray | None  # image -> court (filtered)
    label: str  # accepted | uncertain | rejected
    reasons: list[str]
    raw: HomographySolution | None
    params: CameraParams | None
    source: str  # landmark | flow | predict | none


class CourtCalibrator:
    def __init__(self, image_shape: tuple[int, int], cfg: CalibratorConfig | None = None):
        self.shape = image_shape
        self.cfg = cfg or CalibratorConfig()
        self.cx, self.cy = image_shape[1] / 2, image_shape[0] / 2
        self.flow = FloorFlow()
        self.reset()

    def reset(self) -> None:
        self.cam: PTZCamera | None = None
        self.kf: LinearKalman | None = None
        self.H: np.ndarray | None = None
        self._init_C: list[np.ndarray] = []
        self._init_H: list[np.ndarray] = []
        self.stale = 0
        self.flow.reset()

    # -- helpers --------------------------------------------------------
    def _kf_init(self, p: CameraParams) -> None:
        kf = LinearKalman(p.as_array(), n_pos=4, mode="cv", scale_by_size=False, q_pos=self.cfg.q_angle, q_vel=self.cfg.q_rate, r_meas=0.003, p0_vel=3.0)
        self.kf = kf

    def _measure(self, H_i2c: np.ndarray) -> tuple[CameraParams, float] | None:
        """Fit PTZ parameters to a measured H. Returns (params, noise scale): a worse
        fit is a noisier measurement and gets a proportionally larger R in the filter."""
        assert self.cam is not None
        init = CameraParams.from_array(self.kf.mean) if self.kf is not None else None
        p, rms = self.cam.fit(H_i2c, init)
        if rms > self.cfg.max_fit_rms_px:
            return None
        return p, max(1.0, rms / self.cfg.good_fit_rms_px)

    def _try_fix_camera(self, H_i2c: np.ndarray) -> None:
        dec = decompose_homography(np.linalg.inv(H_i2c), self.cx, self.cy)
        if dec is None or dec.C[2] <= 0.5:
            return
        self._init_C.append(dec.C)
        self._init_H.append(H_i2c)
        if len(self._init_C) >= self.cfg.n_init_frames:
            C = np.median(np.stack(self._init_C), axis=0)
            self.cam = PTZCamera(C, self.cx, self.cy)
            p, _ = self.cam.fit(self._init_H[-1])
            self._kf_init(p)

    # -- main step ------------------------------------------------------
    def step(
        self,
        frame_bgr: np.ndarray | None,
        img_pts: np.ndarray,
        landmark_idx: np.ndarray,
        conf: np.ndarray | None,
        boxes_xyxy: np.ndarray | None = None,
    ) -> CalibrationFrame:
        gcfg = self.cfg.gate
        edges = edges_coarse = None
        if frame_bgr is not None and self.cfg.use_line_support:
            edges = edge_map(frame_bgr)
            cands = solve_homography_candidates(img_pts, landmark_idx, conf, self.shape)
            ok_c = [c for c in cands if c.ok]
            # refine every candidate onto the paint before judging it: a landmark model
            # trained on another league's floor lands close but not on the lines
            if self.cfg.refine_lines and ok_c:
                edges_coarse = edge_map(frame_bgr, ridge_sigma=6.0)
                for c in ok_c:
                    c.H, _ = refine_by_line_support(c.H, frame_bgr, self.cfg.refine_iters, edges_coarse, edges)
            chosen, scores = select_by_line_support(cands, frame_bgr, self.cfg.min_line_support)
            support = max(scores) if scores else 0.0
            raw = chosen if chosen is not None else HomographySolution(
                None, cands[0].inlier_mask, cands[0].residuals_px, cands[0].n_input, cands[0].spread,
                cands[0].reason or "no candidate on the painted lines",
            )
        else:
            support = None
            raw = solve_homography(img_pts, landmark_idx, conf, self.shape)
        g_raw: GateResult = assess(raw, self.shape, prev_H=self.H, cfg=gcfg, line_support=support if raw.ok else None)

        dH = self.flow(frame_bgr, self.H, boxes_xyxy) if frame_bgr is not None else None
        H_flow = None
        if dH is not None and self.H is not None and abs(np.linalg.det(dH)) > 1e-9:
            H_flow = self.H @ np.linalg.inv(dH)
            H_flow /= H_flow[2, 2]
            # the propagated H drifts; snap it back onto the paint when we can see lines
            if edges is not None and self.cfg.refine_lines:
                H_flow, s_flow = refine_by_line_support(H_flow, frame_bgr, self.cfg.refine_iters, edges_coarse, edges)
                if s_flow < self.cfg.min_line_support / 2:
                    H_flow = None  # no visible court lines: do not trust flow either

        # ---- bootstrap: no camera model yet -----------------------------
        if self.cam is None:
            if g_raw.usable:
                # uncertain frames are good enough to seed the camera position (median of
                # several) and to emit coordinates with the uncertain label
                self._try_fix_camera(raw.H)
                self.H = raw.H
                self.stale = 0
                return CalibrationFrame(raw.H, g_raw.label, g_raw.reasons, raw, None, "landmark")
            if H_flow is not None and self.stale < self.cfg.max_stale:
                self.H = H_flow
                self.stale += 1
                return CalibrationFrame(H_flow, "uncertain", ["bootstrap: flow only"], raw, None, "flow")
            self.stale += 1
            return CalibrationFrame(None, "rejected", g_raw.reasons or ["no calibration yet"], raw, None, "none")

        # ---- steady state: Kalman on camera parameters ------------------
        self.kf.predict()
        source = "predict"
        label, reasons = "uncertain", list(g_raw.reasons)
        if g_raw.usable:
            m = self._measure(raw.H)
            if m is not None:
                p, k = m
                self.kf.update(p.as_array(), R=np.diag((k * np.array(self.cfg.r_landmark)) ** 2))
                source, self.stale = "landmark", 0
                label = g_raw.label
            else:
                reasons.append("landmark H inconsistent with PTZ camera")
        if source == "predict" and H_flow is not None:
            m = self._measure(H_flow)
            if m is not None:
                p, k = m
                self.kf.update(p.as_array(), R=np.diag((k * np.array(self.cfg.r_flow)) ** 2))
                source = "flow"
                self.stale += 1
                reasons.append("flow-propagated")
        if source == "predict":
            self.stale += 1
            reasons.append("no measurement")
        params = CameraParams.from_array(self.kf.mean)
        H = self.cam.H_image_to_court(params)
        if self.stale > self.cfg.max_stale:
            self.H = None
            return CalibrationFrame(None, "rejected", reasons + [f"stale {self.stale} frames"], raw, params, source)
        # final validation of the *filtered* H: on the painted lines when we can see them,
        # otherwise against this frame's landmarks
        if edges is not None:
            s_f = line_support(H, edges)
            if s_f >= gcfg.accept_line_support and source == "landmark":
                label = "accepted"
            elif s_f < gcfg.min_line_support / 2:
                label = "rejected"
                reasons.append(f"filtered H not on the paint ({s_f:.3f})")
            elif s_f < gcfg.min_line_support and label == "accepted":
                label = "uncertain"
                reasons.append(f"filtered H weak on the paint ({s_f:.3f})")
        elif raw.ok and raw.n_inliers >= 4:
            idx = np.flatnonzero(raw.inlier_mask)
            from .geometry import LANDMARK_XY

            err = np.linalg.norm(court_to_image(H, LANDMARK_XY[np.asarray(landmark_idx)[idx]]) - np.asarray(img_pts)[idx], axis=1).mean()
            if err > 2 * gcfg.max_mean_err_px_uncertain:
                label = "rejected"
                reasons.append(f"filtered H off landmarks by {err:.0f}px")
            elif err > gcfg.max_mean_err_px and label == "accepted":
                label = "uncertain"
                reasons.append(f"filtered H {err:.1f}px from landmarks")
        self.H = H if label != "rejected" else self.H
        return CalibrationFrame(H if label != "rejected" else None, label, reasons, raw, params, source)
