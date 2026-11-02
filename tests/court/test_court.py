"""Court calibration tests with a synthetic broadcast camera.

We build a realistic pinhole camera (high on one sideline, looking down at the
court), project the true landmarks, corrupt them with noise and outliers, and
check that the solver recovers the court positions and the gate behaves.
"""
import numpy as np
import pytest

from hoops.court import (
    LANDMARK_IDS,
    LANDMARK_XY,
    GateConfig,
    assess,
    court_lines,
    court_to_image,
    foot_points_to_court,
    image_to_court,
    mirror_id,
    solve_homography,
)
from hoops.court.geometry import HALF_L, HALF_W, THREE_BREAK_X, THREE_CORNER_Y, BASKET_X, THREE_R

IMG = (1080, 1920)


def broadcast_camera(pan_deg=0.0, f_px=1800.0, cam_pos=(0.0, -32.0, 14.0)):
    """3x4 projection matrix for a camera high on one sideline aiming at the court.

    pan_deg swings the aim point along the court length (0 = centre court).
    Look-at construction: camera z = forward, x = right, y = down.
    """
    C = np.asarray(cam_pos, dtype=float)
    aim = np.array([np.tan(np.deg2rad(pan_deg)) * abs(C[1]), 0.0, 0.0])
    fwd = aim - C
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ C
    K = np.array([[f_px, 0, IMG[1] / 2], [0, f_px, IMG[0] / 2], [0, 0, 1]])
    return K @ np.hstack([R, t[:, None]])


def project_floor(P, xy):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    X = np.hstack([xy, np.zeros((len(xy), 1)), np.ones((len(xy), 1))])
    p = X @ P.T
    return p[:, :2] / p[:, 2:3]


def visible(px, margin=0):
    return (px[:, 0] >= -margin) & (px[:, 0] < IMG[1] + margin) & (px[:, 1] >= -margin) & (px[:, 1] < IMG[0] + margin)


def test_geometry_sanity():
    assert len(LANDMARK_IDS) == len(set(LANDMARK_IDS))
    assert len(LANDMARK_IDS) >= 30
    # three-point break point is on the arc and on the corner line
    d = np.hypot(THREE_BREAK_X - BASKET_X, THREE_CORNER_Y)
    assert abs(d - THREE_R) < 1e-9
    assert abs(THREE_BREAK_X - (HALF_L - 14.2 * 0.3048)) < 0.02  # ~14.2 ft from baseline
    for i in LANDMARK_IDS:
        m = mirror_id(i)
        assert mirror_id(m) == i
        a, b = LANDMARK_XY[LANDMARK_IDS.index(i)], LANDMARK_XY[LANDMARK_IDS.index(m)]
        assert abs(a[0] + b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
    assert {"sideline_T", "three_arc_L", "centre_circle"} <= set(court_lines())
    assert (np.abs(LANDMARK_XY[:, 0]) <= HALF_L + 1e-9).all() and (np.abs(LANDMARK_XY[:, 1]) <= HALF_W + 1e-9).all()


@pytest.mark.parametrize("pan", [-20.0, 0.0, 25.0])
def test_recover_homography_with_noise_and_outliers(pan):
    rng = np.random.default_rng(abs(int(pan)) + 7)
    P = broadcast_camera(pan_deg=pan)
    px = project_floor(P, LANDMARK_XY)
    vis = visible(px)
    idx = np.flatnonzero(vis)
    assert len(idx) >= 6
    noisy = px[idx] + rng.normal(0, 2.0, (len(idx), 2))
    # corrupt 30% with gross errors (wrong landmark / hallucination)
    n_bad = max(1, int(0.3 * len(idx)))
    bad = rng.choice(len(idx), n_bad, replace=False)
    noisy[bad] += rng.uniform(-300, 300, (n_bad, 2))
    conf = rng.uniform(0.5, 1.0, len(idx))

    sol = solve_homography(noisy, idx, conf, IMG, reproj_px=8.0)
    assert sol.ok, sol.reason
    # no gross outlier may be accepted, and most clean landmarks must be kept
    assert not sol.inlier_mask[bad].any()
    assert sol.n_inliers >= 0.8 * (len(idx) - n_bad)
    # recovered court positions of clean landmarks within ~0.3 m
    good = np.setdiff1d(np.arange(len(idx)), bad)
    rec = image_to_court(sol.H, px[idx][good])
    err_m = np.linalg.norm(rec - LANDMARK_XY[idx][good], axis=1)
    assert np.median(err_m) < 0.3, np.median(err_m)
    gate = assess(sol, IMG)
    assert gate.accepted, gate.reasons


def test_gate_rejects_too_few_or_collinear():
    P = broadcast_camera()
    # three landmarks along one sideline: collinear -> rejected
    ids = [LANDMARK_IDS.index(i) for i in ("LT_corner", "T_half_side", "RT_corner")]
    px = project_floor(P, LANDMARK_XY[ids])
    sol = solve_homography(px, np.array(ids), None, IMG)
    assert not sol.ok
    assert assess(sol, IMG).label == "rejected"


def test_gate_rejects_mirrored_solution():
    P = broadcast_camera()
    px = project_floor(P, LANDMARK_XY)
    idx = np.flatnonzero(visible(px))
    sol = solve_homography(px[idx], idx, None, IMG)
    assert sol.ok and assess(sol, IMG).accepted
    good_H = sol.H.copy()
    # a left/right confusion is a reflection of the court: the camera would be below the
    # floor, so it is rejected absolutely, with or without a previous frame
    flip = np.diag([-1.0, 1.0, 1.0])
    sol.H = flip @ good_H
    g = assess(sol, IMG)
    assert g.label == "rejected" and any("mirrored" in r for r in g.reasons), g.reasons
    assert assess(sol, IMG, prev_H=good_H).label == "rejected"
    # the same camera on the opposite sideline (180 degree rotation) is physically valid
    rot = np.diag([-1.0, -1.0, 1.0])
    sol.H = rot @ good_H
    assert not any("mirrored" in r for r in assess(sol, IMG).reasons)


def test_solver_recovers_from_mirrored_landmark_identities():
    """Simulate the keypoint model's failure mode: the visible basket's landmarks are
    reported with BOTH the correct ids and their left/right mirror ids at the same
    pixels. The solver must return the real (non-mirrored) homography."""
    P = broadcast_camera(pan_deg=20.0)
    px = project_floor(P, LANDMARK_XY)
    idx = np.flatnonzero(visible(px))
    from hoops.court import mirror_id

    mirrored = np.array([LANDMARK_IDS.index(mirror_id(LANDMARK_IDS[i])) for i in idx])
    pts = np.vstack([px[idx], px[idx]])
    ids = np.concatenate([idx, mirrored])
    conf = np.concatenate([np.full(len(idx), 0.5), np.full(len(idx), 0.6)])  # mirrored ones even more confident
    sol = solve_homography(pts, ids, conf, IMG)
    assert sol.ok, sol.reason
    rec = image_to_court(sol.H, px[idx])
    assert np.median(np.linalg.norm(rec - LANDMARK_XY[idx], axis=1)) < 0.05
    assert assess(sol, IMG).accepted


def test_gate_flags_camera_jump_and_disagreement():
    P1 = broadcast_camera(pan_deg=0.0)
    P2 = broadcast_camera(pan_deg=30.0)
    sols = []
    for P in (P1, P2):
        px = project_floor(P, LANDMARK_XY)
        idx = np.flatnonzero(visible(px))
        sols.append(solve_homography(px[idx], idx, None, IMG))
    g = assess(sols[1], IMG, prev_H=sols[0].H)
    assert g.label != "accepted"
    assert any("jump" in r or "disagrees" in r for r in g.reasons), g.reasons
    # the same camera one frame later agrees with itself
    g2 = assess(sols[0], IMG, prev_H=sols[0].H)
    assert g2.accepted


def test_foot_points_round_trip():
    P = broadcast_camera()
    px = project_floor(P, LANDMARK_XY)
    idx = np.flatnonzero(visible(px))
    sol = solve_homography(px[idx], idx, None, IMG)
    feet_court = np.array([[0.0, 0.0], [5.0, -3.0], [-10.0, 6.0]])
    feet_px = court_to_image(sol.H, feet_court)
    boxes = np.stack([feet_px[:, 0] - 30, feet_px[:, 1] - 150, feet_px[:, 0] + 30, feet_px[:, 1]], axis=1)
    rec = foot_points_to_court(sol.H, boxes)
    np.testing.assert_allclose(rec, feet_court, atol=1e-3)
