"""Join tracking-derived features into the college rating model.

Input: court-coordinate parquet files produced by `hoops calibrate --tracks` (one per
game clip or game), tagged with the schedule game_id and which team is which. Output: a
per-team, per-game feature table, then rolling per-team averages that enter the margin
and total models as extra regressors (ridge, fitted alongside the ratings).

Features per team per game (from hoops.features.tracking):
  pace_proxy   : possessions per minute estimated from team-centroid half-court crossings
  spacing      : mean convex-hull area of the five on-court players (m^2)
  speed_mean   : mean player speed (m/s) across the game
  frontcourt   : share of frames with >= 4 players in the attacking half (half-court sets)
These are deliberately simple; the test is whether they move closing-line value, not
whether they are clever.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hoops.features.tracking import rolling_features, validate

FEATURES = ["pace_proxy", "spacing", "speed_mean", "frontcourt"]


def _crossings(x: np.ndarray, t: np.ndarray) -> float:
    """Half-court crossings per minute of a team centroid's x trajectory."""
    if len(x) < 3:
        return 0.0
    s = np.sign(x)
    changes = np.count_nonzero(np.diff(s) != 0)
    minutes = max((t[-1] - t[0]) / 60.0, 1e-6)
    return changes / minutes


def team_game_features(tf: pd.DataFrame) -> pd.DataFrame:
    """tf: TrackingFrame rows (frame, t_s, team in {0,1,-1}, track_id, x_m, y_m) for ONE game.
    Returns one row per team (0/1) with FEATURES."""
    tf = validate(tf)
    rf = rolling_features(tf, window_s=5.0)
    out = []
    for team in (0, 1):
        g = tf[tf.team == team]
        if g.empty:
            out.append(dict(team=team, **{f: np.nan for f in FEATURES}))
            continue
        cen = g.groupby("frame").agg(x=("x_m", "mean"), t=("t_s", "first")).sort_values("t")
        hull = rf[f"team{team}_hull_area"].dropna()
        speed = rf[f"team{team}_speed_mean"].dropna()
        # frontcourt share: >= 4 players on the side where the team's centroid is attacking
        per_frame = g.groupby("frame").x_m.apply(lambda v: (np.sign(v) == np.sign(v.mean())).sum() >= 4)
        out.append(
            dict(
                team=team,
                pace_proxy=_crossings(cen.x.to_numpy(), cen.t.to_numpy()),
                spacing=float(hull.mean()) if len(hull) else np.nan,
                speed_mean=float(speed.mean()) if len(speed) else np.nan,
                frontcourt=float(per_frame.mean()) if len(per_frame) else np.nan,
            )
        )
    return pd.DataFrame(out)


def rolling_team_features(game_feats: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """game_feats: rows (game_id, game_date, team_id, FEATURES...). Returns the same rows with
    FEATURES replaced by the trailing mean over the team's previous `window` games
    (strictly before the game: no look-ahead)."""
    df = game_feats.sort_values(["team_id", "game_date"]).copy()
    for f in FEATURES:
        df[f] = df.groupby("team_id")[f].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return df


def feature_matrix(games: pd.DataFrame, team_feats: pd.DataFrame) -> pd.DataFrame:
    """Per game: home minus away rolling features (NaN -> 0 = 'no information')."""
    tf = team_feats.set_index(["game_id", "team_id"])[FEATURES]
    rows = []
    for r in games.itertuples():
        h = tf.loc[(r.game_id, r.home_id)] if (r.game_id, r.home_id) in tf.index else None
        a = tf.loc[(r.game_id, r.away_id)] if (r.game_id, r.away_id) in tf.index else None
        diff = {f: (float(h[f]) if h is not None and pd.notna(h[f]) else 0.0) - (float(a[f]) if a is not None and pd.notna(a[f]) else 0.0) for f in FEATURES}
        rows.append(dict(game_id=r.game_id, **{f"d_{f}": v for f, v in diff.items()}))
    return pd.DataFrame(rows)


def ridge_adjustment(X: np.ndarray, resid: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Fit feature coefficients to the rating model's margin residuals (standardised X)."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    return np.linalg.solve(A, Xs.T @ resid)
