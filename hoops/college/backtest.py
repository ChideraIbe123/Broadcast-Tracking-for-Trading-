"""Walk-forward backtest of the rating model: for every game day of a season, fit on
everything before that day (plus last season's ratings as a prior) and predict that day.

Reports Brier, log loss, margin MAE and calibration, overall and by month, against a
naive "home team wins X%" reference. `calibrate()` fits a margin shrink factor and a
probability sigma on one season's walk-forward predictions for use on the next; the
raw model is overconfident (ridge ratings are too spread for out-of-sample games).
There are no historical Kalshi closes offline, so the market comparison happens live
through the paper ledger (closing-line value).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import non_d1_teams
from .ratings import RatingModel, fit_ratings


def season_prior(games: pd.DataFrame, season: int, prior_tau: float = 400.0, prior_cap: float = 35.0) -> RatingModel | None:
    prev = games[games.season == season - 1]
    if prev.completed.sum() < 500:
        return None
    return fit_ratings(prev, non_d1=non_d1_teams(games), tau_days=prior_tau, cap=prior_cap)


def walk_forward(games: pd.DataFrame, season: int, refit_every_days: int = 1, scale: float = 1.0, sigma: float | None = None, prior_tau: float = 400.0, prior_cap: float = 35.0, **fit_kw) -> pd.DataFrame:
    g = games[(games.season == season) & games.completed].copy()
    prior = season_prior(games, season, prior_tau, prior_cap)
    nd1 = non_d1_teams(games)
    days = sorted(g.game_date.dt.normalize().unique())
    rows = []
    model = None
    last_fit = None
    for day in days:
        if model is None or last_fit is None or (day - last_fit).days >= refit_every_days:
            model = fit_ratings(games[games.season == season], asof=day, prior=prior, non_d1=nd1, **fit_kw)
            model.scale = scale
            if sigma is not None:
                model.sigma = sigma
            last_fit = day
        todays = g[g.game_date.dt.normalize() == day]
        for r in todays.itertuples():
            rows.append(
                dict(
                    game_id=r.game_id, date=day, home=r.home_display_name, away=r.away_display_name,
                    p_home=model.p_home(r.home_id, r.away_id, r.neutral), pred_margin=model.margin(r.home_id, r.away_id, r.neutral), margin=r.margin,
                    pred_total=model.total(r.home_id, r.away_id), total=r.total,
                    home_win=int(r.margin > 0), n_games_fit=model.n_games, sigma=model.sigma,
                    non_d1=int(r.home_id in nd1 or r.away_id in nd1),
                )
            )
    return pd.DataFrame(rows)


def calibrate(pred: pd.DataFrame) -> tuple[float, float]:
    """Best (scale, sigma) by log loss on walk-forward predictions made with scale 1.
    Returns the shrink factor for predicted margins and the probability sigma."""
    y = pred.home_win.to_numpy(float)
    raw = pred.pred_margin.to_numpy(float)  # produced with scale 1
    best = (1e9, 1.0, 11.0)
    for scale in np.arange(0.6, 1.21, 0.05):
        for sigma in np.arange(9.0, 16.1, 0.5):
            p = np.clip(norm.cdf(scale * raw / sigma), 1e-4, 1 - 1e-4)
            ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
            if ll < best[0]:
                best = (ll, float(scale), float(sigma))
    return best[1], best[2]


def summarise(pred: pd.DataFrame) -> dict:
    y, p = pred.home_win.to_numpy(float), pred.p_home.clip(1e-4, 1 - 1e-4).to_numpy()
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    base = float(y.mean())
    ll_naive = float(-np.mean(y * np.log(base) + (1 - y) * np.log(1 - base)))
    out = {
        "games": int(len(pred)),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_naive": float(np.mean((base - y) ** 2)),
        "log_loss": ll,
        "log_loss_naive": ll_naive,
        "accuracy": float(np.mean((p > 0.5) == (y > 0.5))),
        "margin_mae": float(np.mean(np.abs(pred.pred_margin - pred.margin))),
        "total_mae": float(np.mean(np.abs(pred.pred_total - pred.total))),
        "home_win_rate": base,
    }
    if "non_d1" in pred:
        d1 = pred[pred.non_d1 == 0]
        out["d1_only"] = dict(games=int(len(d1)), brier=float(np.mean((d1.p_home - d1.home_win) ** 2)), margin_mae=float(np.mean(np.abs(d1.pred_margin - d1.margin))))
    pred = pred.copy()
    pred["month"] = pred.date.dt.strftime("%Y-%m")
    out["by_month"] = {
        m: dict(games=int(len(d)), brier=round(float(np.mean((d.p_home - d.home_win) ** 2)), 4), margin_mae=round(float(np.mean(np.abs(d.pred_margin - d.margin))), 2))
        for m, d in pred.groupby("month")
    }
    bins = np.clip((p * 10).astype(int), 0, 9)
    out["calibration"] = {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": (round(float(p[bins == b].mean()), 3), round(float(y[bins == b].mean()), 3), int((bins == b).sum())) for b in range(10) if (bins == b).any()}
    return out
