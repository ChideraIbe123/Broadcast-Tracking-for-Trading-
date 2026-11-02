"""Line support must prefer the homography whose court drawing lies on the painted
lines over an internally consistent 'ghost' of the other end of the court."""
import cv2
import numpy as np

from hoops.court.geometry import LANDMARK_XY, MIRROR_INDEX, court_lines
from hoops.court.homography import court_to_image, solve_homography_candidates
from hoops.court.lines import edge_map, line_support, select_by_line_support
from hoops.court.camera import PTZCamera, CameraParams

IMG = (720, 1280)


def render(H_i2c: np.ndarray) -> np.ndarray:
    """Dark floor with white court lines drawn through the true homography, plus noise."""
    rng = np.random.default_rng(0)
    img = np.full((IMG[0], IMG[1], 3), 60, np.uint8)
    img = (img + rng.integers(0, 25, img.shape, dtype=np.uint8)).astype(np.uint8)
    for pts in court_lines().values():
        px = court_to_image(H_i2c, pts)
        ok = np.isfinite(px).all(1) & (np.abs(px[:, 0]) < 5000) & (np.abs(px[:, 1]) < 5000)
        if ok.sum() >= 2:
            cv2.polylines(img, [px[ok].astype(np.int32).reshape(-1, 1, 2)], False, (235, 235, 235), 3, cv2.LINE_AA)
    return img


def test_line_support_prefers_true_homography_over_ghost():
    cam = PTZCamera(np.array([1.0, -30.0, 12.0]), IMG[1] / 2, IMG[0] / 2)
    H_true = cam.H_image_to_court(CameraParams(0.35, 0.33, 0.0, np.log(1600.0)))  # right basket in view
    img = render(H_true)
    edges = edge_map(img)
    # a ghost: the same frame interpreted as the LEFT basket seen by a camera that
    # panned the other way (physically valid, lines nowhere near the paint)
    H_ghost = cam.H_image_to_court(CameraParams(-0.35, 0.33, 0.0, np.log(1600.0)))
    s_true, s_ghost = line_support(H_true, edges), line_support(H_ghost, edges)
    assert s_true > 0.1, s_true
    assert s_ghost < s_true / 3, (s_true, s_ghost)


def test_refinement_improves_perturbed_homography_and_never_crashes():
    from hoops.court.lines import refine_by_line_support

    cam = PTZCamera(np.array([1.0, -30.0, 12.0]), IMG[1] / 2, IMG[0] / 2)
    p_true = CameraParams(0.3, 0.33, 0.0, np.log(1600.0))
    img = render(cam.H_image_to_court(p_true))
    # a few pixels off: the kind of bias a landmark model trained elsewhere produces
    p_bad = CameraParams(0.3 + 0.006, 0.33 - 0.004, 0.002, np.log(1600.0) + 0.015)
    H_bad = cam.H_image_to_court(p_bad)
    e = edge_map(img)
    s_bad = line_support(H_bad, e)
    H_ref, s_ref = refine_by_line_support(H_bad, img)
    assert s_ref > s_bad
    px = court_to_image(cam.H_image_to_court(p_true), LANDMARK_XY)
    err_bad = np.nanmedian(np.linalg.norm(court_to_image(H_bad, LANDMARK_XY) - px, axis=1))
    err_ref = np.nanmedian(np.linalg.norm(court_to_image(H_ref, LANDMARK_XY) - px, axis=1))
    assert err_ref < err_bad / 2, (err_bad, err_ref)
    # degenerate input must not raise
    H_junk = np.array([[1e-9, 0, 0], [0, 1e-9, 0], [0, 0, 1.0]])
    H_out, _ = refine_by_line_support(H_junk, img)
    assert H_out is H_junk or np.isfinite(H_out).all()


def test_gate_rejects_implausible_cameras_and_fan_homographies():
    from hoops.court.gate import assess
    from hoops.court.homography import HomographySolution

    def sol(H):
        return HomographySolution(H, np.ones(8, bool), np.ones(8), 8, 0.1)

    good = PTZCamera(np.array([1.0, -30.0, 12.0]), IMG[1] / 2, IMG[0] / 2).H_image_to_court(CameraParams(0.3, 0.33, 0.0, np.log(1600.0)))
    assert assess(sol(good), IMG, line_support=0.2).accepted
    # camera 1 m above the floor: not a broadcast camera
    low = PTZCamera(np.array([1.0, -30.0, 1.0]), IMG[1] / 2, IMG[0] / 2).H_image_to_court(CameraParams(0.3, 0.05, 0.0, np.log(1600.0)))
    g = assess(sol(low), IMG, line_support=0.2)
    # either physical check may fire first (a 1 m camera also flattens the court to a sliver)
    assert g.label == "rejected" and any(("height" in r) or ("sliver" in r) for r in g.reasons), g.reasons
    # absurd focal length (fisheye-like) is rejected too
    wide = PTZCamera(np.array([1.0, -30.0, 12.0]), IMG[1] / 2, IMG[0] / 2).H_image_to_court(CameraParams(0.3, 0.33, 0.0, np.log(200.0)))
    g = assess(sol(wide), IMG, line_support=0.2)
    assert g.label == "rejected" and any("focal" in r for r in g.reasons), g.reasons
    # a collapsed "fan" homography gets no line-support credit
    img = render(good)
    fan = good.copy()
    fan[2, :] = [0.002, 0.002, 1.0]  # horizon through the image: lines converge
    assert line_support(fan, edge_map(img)) < 0.5 * line_support(good, edge_map(img))


def test_candidates_and_selection_recover_real_court_from_mixed_landmarks():
    cam = PTZCamera(np.array([1.0, -30.0, 12.0]), IMG[1] / 2, IMG[0] / 2)
    H_true = cam.H_image_to_court(CameraParams(0.35, 0.33, 0.0, np.log(1600.0)))
    img = render(H_true)
    px = court_to_image(H_true, LANDMARK_XY)
    vis = (px[:, 0] > 0) & (px[:, 0] < IMG[1]) & (px[:, 1] > 0) & (px[:, 1] < IMG[0])
    real_idx = np.flatnonzero(vis)
    # ghost: a complete, self-consistent set of the OTHER end's landmarks drawn where a
    # left-basket view would put them (different pixels, same handedness)
    H_ghost = cam.H_image_to_court(CameraParams(-0.35, 0.33, 0.0, np.log(1600.0)))
    ghost_idx = MIRROR_INDEX[real_idx]
    ghost_px = court_to_image(H_ghost, LANDMARK_XY[ghost_idx])
    pts = np.vstack([px[real_idx], ghost_px])
    ids = np.concatenate([real_idx, ghost_idx])
    conf = np.concatenate([np.full(len(real_idx), 0.45), np.full(len(ghost_idx), 0.5)])
    cands = solve_homography_candidates(pts, ids, conf, IMG, n_candidates=3)
    assert len(cands) >= 2 and all(c.ok for c in cands[:2])
    chosen, scores = select_by_line_support(cands, img)
    assert chosen is not None
    rec = court_to_image(chosen.H, LANDMARK_XY[real_idx])
    assert np.median(np.linalg.norm(rec - px[real_idx], axis=1)) < 2.0, scores
