"""Court specs: same landmark ids for every league, consistent geometry, clean switching."""
import numpy as np
import pytest

from hoops.court import geometry as g


@pytest.fixture(autouse=True)
def restore_nba():
    yield
    g.set_court("nba")


@pytest.mark.parametrize("name", list(g.COURTS))
def test_spec_geometry_consistent(name):
    c = g.COURTS[name]
    lms = g.build_landmarks(c)
    assert [lm.id for lm in lms] == g.LANDMARK_IDS  # identical id order across leagues
    xy = np.array([lm.xy for lm in lms])
    fin = np.isfinite(xy).all(1)
    assert (np.abs(xy[fin, 0]) <= c.half_l + 1e-9).all() and (np.abs(xy[fin, 1]) <= c.half_w + 1e-9).all()
    # three-point break (when present) lies on the arc and on the straight corner line
    if c.three_break_x is not None:
        assert abs(np.hypot(c.three_break_x - c.basket_x, c.three_corner_y) - c.three_r) < 1e-9
    else:
        assert np.isnan(xy[g.landmark_index("LT_three_break")]).any()  # x is NaN => unusable
        # arc meets the baseline: distance from basket to (baseline, corner_y) equals the radius
        assert abs(np.hypot(c.basket_from_baseline, c.three_corner_y) - c.three_r) < 1e-9
    lines = g.build_lines(c)
    assert {"sideline_T", "three_arc_L", "paint_R", "centre_circle"} <= set(lines)


def test_known_dimensions():
    nba, ncaa, fiba, old = g.COURTS["nba"], g.COURTS["ncaa_men"], g.COURTS["fiba"], g.COURTS["ncaa_men_pre2019"]
    assert abs(nba.three_r - 7.239) < 1e-3 and abs(nba.paint_half_width * 2 - 4.8768) < 1e-3
    assert abs(ncaa.three_r - 6.7501) < 1e-3 and abs(ncaa.paint_half_width * 2 - 3.6576) < 1e-3
    assert abs(ncaa.three_corner_y - (21 + 7.875 / 12) * g.FT) < 1e-9  # corner three measured from the basket
    assert abs(fiba.three_corner_y - 6.6) < 1e-9 and abs(fiba.ft_x - 8.2) < 1e-9 and abs(fiba.basket_x - 12.425) < 1e-9
    assert old.three_break_x is None and abs(old.three_r - 6.3246) < 1e-3


def test_set_court_updates_shared_arrays_in_place():
    ref = g.LANDMARK_XY  # the array other modules imported
    i = g.landmark_index("R_three_apex")
    x_nba = ref[i, 0]
    g.set_court("fiba")
    assert g.cur().name == "fiba"
    assert ref[i, 0] != x_nba and abs(ref[i, 0] - (12.425 - 6.75)) < 1e-9
    assert g.court_lines()["sideline_T"][0, 1] == 7.5
    g.set_court("nba")
    assert ref[i, 0] == x_nba


def test_solver_skips_absent_landmarks():
    from hoops.court.homography import solve_homography

    g.set_court("ncaa_men_pre2019")
    # four real corners + a bogus "three_break" (absent on this court) with wild pixels
    ids = [g.landmark_index(n) for n in ("LT_corner", "LB_corner", "RT_corner", "RB_corner", "LT_three_break")]
    px = np.array([[100, 100], [100, 600], [1200, 100], [1200, 600], [5, 5]], float)
    sol = solve_homography(px, np.array(ids), None, (720, 1280), require_orientation=False)
    assert sol.ok and not sol.inlier_mask[-1]
