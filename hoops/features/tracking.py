"""Tracking features on the shared `TrackingFrame` schema.

One schema for two sources so the same code trains (on 2015-16 SportVU) and
serves (on our broadcast tracking output):

    frame | t_s | team (0/1/-1 for ball/unknown) | track_id | x_m | y_m [| z_m]

Court frame: centre origin, x along the length (metres). Features are computed
per frame and then aggregated over rolling windows or possessions by the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from hoops.court.geometry import cur

REQUIRED = ("frame", "t_s", "team", "track_id", "x_m", "y_m")


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"TrackingFrame missing columns {missing}")
    return df


def _hull_area(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)  # 2-D 'volume' is the area
    except Exception:
        return 0.0


def spacing(xy: np.ndarray) -> dict[str, float]:
    """Team shape descriptors from (N, 2) player positions."""
    if len(xy) == 0:
        return {"hull_area": 0.0, "spread": 0.0, "width": 0.0, "depth": 0.0}
    c = xy.mean(axis=0)
    return {
        "hull_area": _hull_area(xy),
        "spread": float(np.linalg.norm(xy - c, axis=1).mean()),
        "width": float(np.ptp(xy[:, 1])) if len(xy) > 1 else 0.0,
        "depth": float(np.ptp(xy[:, 0])) if len(xy) > 1 else 0.0,
    }


def nearest_defender(off_xy: np.ndarray, def_xy: np.ndarray) -> np.ndarray:
    """Distance from each offensive player to the closest defender (metres)."""
    if len(off_xy) == 0 or len(def_xy) == 0:
        return np.full(len(off_xy), np.nan)
    d = np.linalg.norm(off_xy[:, None, :] - def_xy[None, :, :], axis=2)
    return d.min(axis=1)


def velocities(df: pd.DataFrame) -> pd.DataFrame:
    """Per-track speed/acceleration via central differences (m/s, m/s^2)."""
    df = validate(df).sort_values(["track_id", "frame"]).copy()
    g = df.groupby("track_id")
    dt = g["t_s"].diff()
    vx, vy = g["x_m"].diff() / dt, g["y_m"].diff() / dt
    df["speed"] = np.hypot(vx, vy)
    df["accel"] = df.groupby("track_id")["speed"].diff() / dt
    return df


def ball_owner(frame: pd.DataFrame, max_dist: float = 1.2) -> int | None:
    """Track id of the player closest to the ball, if within max_dist metres."""
    ball = frame[frame.team == -1]
    players = frame[frame.team >= 0]
    if ball.empty or players.empty:
        return None
    b = ball[["x_m", "y_m"]].to_numpy()[0]
    d = np.linalg.norm(players[["x_m", "y_m"]].to_numpy() - b, axis=1)
    i = int(d.argmin())
    return int(players.track_id.iloc[i]) if d[i] <= max_dist else None


def attacking_basket_x(team_xy: np.ndarray) -> float:
    """Which basket a team is attacking this frame: the one on the side where most of them are."""
    bx = cur().basket_x
    return bx if np.mean(team_xy[:, 0] > 0) >= 0.5 else -bx


def frame_features(frame: pd.DataFrame, offence: int | None = None) -> dict[str, float]:
    """All per-frame features for one frame of TrackingFrame rows."""
    out: dict[str, float] = {}
    teams = {t: frame[frame.team == t][["x_m", "y_m"]].to_numpy() for t in (0, 1)}
    for t, xy in teams.items():
        for k, v in spacing(xy).items():
            out[f"team{t}_{k}"] = v
    if offence is None:
        owner = ball_owner(frame)
        if owner is not None:
            offence = int(frame[frame.track_id == owner].team.iloc[0])
    out["offence"] = -1 if offence is None else offence
    if offence in (0, 1):
        off, de = teams[offence], teams[1 - offence]
        nd = nearest_defender(off, de)
        out["nd_mean"] = float(np.nanmean(nd)) if len(nd) else np.nan
        out["nd_min"] = float(np.nanmin(nd)) if len(nd) else np.nan
        if len(off):
            bx = attacking_basket_x(off)
            dist_basket = np.linalg.norm(off - np.array([bx, 0.0]), axis=1)
            out["off_dist_basket_mean"] = float(dist_basket.mean())
            out["off_in_frontcourt"] = float(np.mean(np.sign(off[:, 0]) == np.sign(bx)))
        ball = frame[frame.team == -1]
        if not ball.empty and len(off):
            b = ball[["x_m", "y_m"]].to_numpy()[0]
            out["ball_dist_basket"] = float(np.linalg.norm(b - np.array([attacking_basket_x(off), 0.0])))
            out["ball_x_norm"] = float(b[0] / cur().half_l)
    return out


def rolling_features(df: pd.DataFrame, window_s: float = 5.0) -> pd.DataFrame:
    """Per-frame features + rolling means over the trailing window + team speeds."""
    df = velocities(validate(df))
    rows = []
    for f, g in df.groupby("frame"):
        feats = frame_features(g)
        feats["frame"] = f
        feats["t_s"] = float(g.t_s.iloc[0])
        for t in (0, 1):
            s = g[g.team == t].speed
            feats[f"team{t}_speed_mean"] = float(s.mean()) if len(s) else np.nan
        rows.append(feats)
    out = pd.DataFrame(rows).sort_values("t_s")
    num = [c for c in out.columns if c not in ("frame", "t_s", "offence")]
    if len(out):
        fps = 1.0 / max(np.median(np.diff(out.t_s)), 1e-6) if len(out) > 1 else 25.0
        w = max(int(window_s * fps), 1)
        roll = out[num].rolling(w, min_periods=1).mean().add_suffix(f"_r{int(window_s)}s")
        out = pd.concat([out, roll], axis=1)
    return out
