import numpy as np
import pandas as pd

from hoops.college.video_features import FEATURES, feature_matrix, ridge_adjustment, rolling_team_features, team_game_features


def synth_game(fps=25, seconds=60, pace_switches=12):
    rows = []
    rng = np.random.default_rng(0)
    n = fps * seconds
    for f in range(n):
        t = f / fps
        side = 1 if int(t / (seconds / pace_switches)) % 2 == 0 else -1  # possession flips
        for team in (0, 1):
            base = side * 8 if team == 0 else side * 6
            for k in range(5):
                rows.append(dict(frame=f, t_s=t, team=team, track_id=team * 10 + k, x_m=base + rng.normal(0, 2), y_m=rng.normal(0, 3)))
    return pd.DataFrame(rows)


def test_team_game_features_shape_and_pace():
    tf = synth_game()
    feats = team_game_features(tf)
    assert list(feats.team) == [0, 1] and set(FEATURES) <= set(feats.columns)
    # 12 possession flips in 60 s -> ~11 crossings/min for each team centroid
    assert 8 < feats.pace_proxy.iloc[0] < 14, feats.pace_proxy.iloc[0]
    assert feats.spacing.iloc[0] > 0 and feats.frontcourt.iloc[0] > 0.9


def test_rolling_has_no_lookahead_and_matrix_joins():
    gf = pd.DataFrame(
        [dict(game_id=i, game_date=pd.Timestamp("2026-11-01") + pd.Timedelta(days=i), team_id="A", pace_proxy=10 + i, spacing=50.0, speed_mean=2.0, frontcourt=0.8) for i in range(4)]
        + [dict(game_id=i, game_date=pd.Timestamp("2026-11-01") + pd.Timedelta(days=i), team_id="B", pace_proxy=20.0, spacing=60.0, speed_mean=2.5, frontcourt=0.7) for i in range(4)]
    )
    r = rolling_team_features(gf, window=2)
    a = r[r.team_id == "A"].sort_values("game_date")
    assert np.isnan(a.pace_proxy.iloc[0])  # nothing before the first game
    assert a.pace_proxy.iloc[1] == 10.0 and a.pace_proxy.iloc[3] == 11.5  # mean of games 1,2 -> (11+12)/2
    games = pd.DataFrame([dict(game_id=3, home_id="A", away_id="B")])
    X = feature_matrix(games, r)
    assert abs(X.d_pace_proxy.iloc[0] - (11.5 - 20.0)) < 1e-9
    assert abs(X.d_spacing.iloc[0] - (50.0 - 60.0)) < 1e-9
    beta = ridge_adjustment(np.random.default_rng(1).normal(size=(100, 4)), np.random.default_rng(2).normal(size=100))
    assert beta.shape == (4,)
