import numpy as np
import pandas as pd

from hoops.court.geometry import BASKET_X
from hoops.features.tracking import ball_owner, frame_features, nearest_defender, rolling_features, spacing


def synth_frames(n=50, fps=25.0):
    rows = []
    rng = np.random.default_rng(0)
    base0 = np.array([[8, 0], [10, 3], [10, -3], [6, 5], [6, -5]], float)  # offence near right basket
    base1 = base0 + np.array([1.0, 0.0])  # defenders a metre closer to the basket
    for f in range(n):
        t = f / fps
        for tid, p in enumerate(base0 + 0.1 * f):
            rows.append(dict(frame=f, t_s=t, team=0, track_id=tid, x_m=p[0], y_m=p[1]))
        for tid, p in enumerate(base1 + 0.1 * f):
            rows.append(dict(frame=f, t_s=t, team=1, track_id=10 + tid, x_m=p[0], y_m=p[1]))
        rows.append(dict(frame=f, t_s=t, team=-1, track_id=99, x_m=8 + 0.1 * f + 0.3, y_m=0.2))
    return pd.DataFrame(rows)


def test_spacing_and_nearest_defender():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    s = spacing(sq)
    assert abs(s["hull_area"] - 4.0) < 1e-9 and s["width"] == 2.0 and s["depth"] == 2.0
    nd = nearest_defender(sq, np.array([[1.0, 1.0]]))
    np.testing.assert_allclose(nd, np.sqrt(2))


def test_frame_features_identify_offence_and_basket():
    df = synth_frames(1)
    fr = df[df.frame == 0]
    assert ball_owner(fr) == 0  # player 0 is 0.36 m from the ball
    f = frame_features(fr)
    assert f["offence"] == 0
    assert 0.9 < f["nd_min"] <= 1.0 + 1e-9  # defenders exactly 1 m away
    assert abs(f["ball_dist_basket"] - np.linalg.norm([8.3 - BASKET_X, 0.2])) < 1e-9
    assert f["off_in_frontcourt"] == 1.0


def test_rolling_features_have_speeds():
    out = rolling_features(synth_frames(50), window_s=1.0)
    assert len(out) == 50
    # everyone drifts (0.1, 0.1) m/frame at 25 fps = 2.5*sqrt(2) m/s
    assert abs(out.team0_speed_mean.iloc[10] - 2.5 * np.sqrt(2)) < 1e-6
    assert "team0_hull_area_r1s" in out.columns
