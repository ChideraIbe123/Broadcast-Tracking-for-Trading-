"""Camera model tests on a synthetic textured broadcast.

We render a textured floor through a panning/zooming PTZ camera so that optical
flow has something to track, feed noisy landmarks (dropped entirely on some
frames) to the calibrator, and check that the filtered homography stays accurate
and continuous, including through the landmark-free stretch.
"""
import numpy as np
import pytest
import cv2

from hoops.court.camera import (
    CameraParams,
    CourtCalibrator,
    PTZCamera,
    decompose_homography,
    ptr_from_rotation,
    rotation_from_ptr,
)
from hoops.court.geometry import HALF_L, HALF_W, LANDMARK_XY
from hoops.court.homography import court_to_image, image_to_court, project

IMG = (720, 1280)
C_TRUE = np.array([2.0, -30.0, 13.0])


def test_rotation_round_trip():
    for pan, tilt, roll in [(0.0, 0.3, 0.0), (0.4, 0.25, 0.02), (-0.6, 0.4, -0.03)]:
        R = rotation_from_ptr(pan, tilt, roll)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) > 0
        p, t, r = ptr_from_rotation(R)
        np.testing.assert_allclose([p, t, r], [pan, tilt, roll], atol=1e-9)


def test_decompose_recovers_pose():
    cam = PTZCamera(C_TRUE, IMG[1] / 2, IMG[0] / 2)
    p = CameraParams(0.2, 0.35, 0.01, np.log(1700.0))
    Hc2i = cam.H_court_to_image(p)
    dec = decompose_homography(Hc2i, cam.cx, cam.cy)
    assert dec is not None
    assert abs(dec.f - 1700.0) / 1700.0 < 1e-6
    np.testing.assert_allclose(dec.C, C_TRUE, atol=1e-6)
    np.testing.assert_allclose([dec.params.pan, dec.params.tilt, dec.params.roll], [0.2, 0.35, 0.01], atol=1e-6)


def test_fit_recovers_params_from_noisy_H():
    cam = PTZCamera(C_TRUE, IMG[1] / 2, IMG[0] / 2)
    p = CameraParams(-0.3, 0.3, 0.0, np.log(1500.0))
    H = cam.H_image_to_court(p)
    fitted, rms = cam.fit(H)
    assert rms < 1e-3
    np.testing.assert_allclose(fitted.as_array(), p.as_array(), atol=1e-5)


class SynthBroadcast:
    """Textured floor rendered through a PTZ camera, with noisy landmarks."""

    def __init__(self, seed=0, n=90):
        self.rng = np.random.default_rng(seed)
        self.cam = PTZCamera(C_TRUE, IMG[1] / 2, IMG[0] / 2)
        self.n = n
        # texture in court space: 40 px per metre
        self.ppm = 40
        tw, th = int((2 * HALF_L + 4) * self.ppm), int((2 * HALF_W + 4) * self.ppm)
        tex = self.rng.integers(60, 200, (th, tw), dtype=np.uint8)
        tex = cv2.GaussianBlur(tex, (0, 0), 1.5)
        for _ in range(400):  # blobs so corners exist at every scale
            x, y = self.rng.integers(0, tw), self.rng.integers(0, th)
            cv2.circle(tex, (int(x), int(y)), int(self.rng.integers(3, 14)), int(self.rng.integers(0, 255)), -1)
        self.T = np.array([[self.ppm, 0, (HALF_L + 2) * self.ppm], [0, -self.ppm, (HALF_W + 2) * self.ppm], [0, 0, 1.0]])  # court->texture px
        # painted court lines, so the line-support check has something to align with
        from hoops.court.geometry import court_lines

        for pts in court_lines().values():
            tp = (np.hstack([pts, np.ones((len(pts), 1))]) @ self.T.T)[:, :2]
            cv2.polylines(tex, [tp.astype(np.int32).reshape(-1, 1, 2)], False, 245, 3, cv2.LINE_AA)
        self.tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)

    def params(self, i):
        t = i / self.n
        pan = -0.35 + 0.7 * t  # sweep left to right
        tilt = 0.32 + 0.02 * np.sin(4 * t)
        logf = np.log(1400 + 500 * t)  # zoom in
        return CameraParams(pan, tilt, 0.0, logf)

    def frame(self, i):
        p = self.params(i)
        Hc2i = self.cam.H_court_to_image(p)
        Hpx = Hc2i @ np.linalg.inv(self.T)  # texture px -> image
        img = cv2.warpPerspective(self.tex, Hpx, (IMG[1], IMG[0]))
        return img, self.cam.H_image_to_court(p), Hc2i

    def landmarks(self, Hc2i, noise=2.0, drop=0.0, outlier_frac=0.2):
        px = project(Hc2i, LANDMARK_XY)
        vis = (px[:, 0] > 0) & (px[:, 0] < IMG[1]) & (px[:, 1] > 0) & (px[:, 1] < IMG[0])
        idx = np.flatnonzero(vis)
        idx = idx[self.rng.random(len(idx)) >= drop]
        pts = px[idx] + self.rng.normal(0, noise, (len(idx), 2))
        n_bad = int(outlier_frac * len(idx))
        if n_bad:
            bad = self.rng.choice(len(idx), n_bad, replace=False)
            pts[bad] += self.rng.uniform(-200, 200, (n_bad, 2))
        return pts, idx, self.rng.uniform(0.5, 1.0, len(idx))


def court_error_m(H_est, H_true_i2c):
    """Mean distance in metres between where H_est and the truth put a grid of pixels."""
    us = np.linspace(100, IMG[1] - 100, 8)
    vs = np.linspace(IMG[0] * 0.45, IMG[0] - 50, 5)
    px = np.array([[u, v] for u in us for v in vs])
    return float(np.linalg.norm(image_to_court(H_est, px) - image_to_court(H_true_i2c, px), axis=1).mean())


def test_calibrator_tracks_pan_zoom_and_survives_landmark_gap():
    sb = SynthBroadcast()
    cal = CourtCalibrator(IMG)
    errs, labels, sources, centres = [], [], [], []
    for i in range(sb.n):
        img, H_true, Hc2i = sb.frame(i)
        if 40 <= i < 52:  # no landmarks at all (replay graphic / close-up region)
            pts, idx, conf = np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
        else:
            pts, idx, conf = sb.landmarks(Hc2i)
        out = cal.step(img, pts, idx, conf, boxes_xyxy=None)
        labels.append(out.label)
        sources.append(out.source)
        errs.append(court_error_m(out.H, H_true) if out.H is not None else np.nan)
        centres.append(court_to_image(out.H, np.zeros((1, 2)))[0] if out.H is not None else None)
    errs = np.array(errs)
    assert cal.cam is not None, "camera position never fixed"
    np.testing.assert_allclose(cal.cam.C, C_TRUE, atol=1.5)
    # steady state: accurate on landmark frames
    steady = errs[15:40]
    assert np.nanmedian(steady) < 0.5, np.nanmedian(steady)
    # during the gap we must still output a usable H carried by flow, with bounded error
    gap = errs[40:52]
    assert all(lbl != "rejected" for lbl in labels[40:52]), labels[40:52]
    assert any(s == "flow" for s in sources[40:52]), sources[40:52]
    assert np.nanmax(gap) < 1.5, gap
    # recovery after the gap
    assert np.nanmedian(errs[60:]) < 0.5
    # continuity: once calibrated, the projected court centre never jumps more than 40 px per frame
    jumps = [np.linalg.norm(b - a) for a, b in zip(centres[15:-1], centres[16:]) if a is not None and b is not None]
    assert jumps and max(jumps) < 40, max(jumps)


def test_calibrator_rejects_when_stale_too_long():
    sb = SynthBroadcast(seed=3, n=60)
    cal = CourtCalibrator(IMG)
    cal.cfg.max_stale = 5
    last = None
    for i in range(sb.n):
        img, H_true, Hc2i = sb.frame(i)
        if i >= 20:
            img = np.zeros_like(img)  # black frames: no flow, no landmarks
            pts, idx, conf = np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
        else:
            pts, idx, conf = sb.landmarks(Hc2i)
        last = cal.step(img, pts, idx, conf)
    assert last.label == "rejected" and last.H is None
