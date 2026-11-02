"""Margin-based team ratings for pre-game probabilities (the baseline the video has to beat).

Model: home_margin = r_home - r_away + hfa * (not neutral) + noise.
Fitted by weighted ridge regression on completed games, with
  * exponential time decay inside the season (recent games count more),
  * shrinkage towards a prior: last season's final ratings, regressed to the mean, so
    the model has an opinion on opening night when the season has three games,
  * blowouts capped so one 50-point game does not dominate.
Win probability = Phi(predicted margin / sigma), sigma from residuals (~11 points).
A second ridge on points for/against gives an expected total.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class RatingModel:
    rating: dict[str, float] = field(default_factory=dict)
    off: dict[str, float] = field(default_factory=dict)  # points scored above average
    dfn: dict[str, float] = field(default_factory=dict)  # points allowed above average (positive = bad)
    hfa: float = 3.2
    sigma: float = 11.0
    base_points: float = 72.0
    n_games: int = 0
    # out-of-sample calibration (fit on a previous season's walk-forward predictions):
    # predicted margins are shrunk by `scale`, probabilities use `sigma`
    scale: float = 1.0

    unknown_rating: float = -10.0  # a team we have never seen is probably a weak non-D-I side

    def margin(self, home: str, away: str, neutral: bool = False) -> float:
        raw = self.rating.get(home, self.unknown_rating) - self.rating.get(away, self.unknown_rating) + (0.0 if neutral else self.hfa)
        return self.scale * raw

    def p_home(self, home: str, away: str, neutral: bool = False) -> float:
        return float(norm.cdf(self.margin(home, away, neutral) / self.sigma))

    def total(self, home: str, away: str) -> float:
        return 2 * self.base_points + self.off.get(home, 0.0) + self.dfn.get(away, 0.0) + self.off.get(away, 0.0) + self.dfn.get(home, 0.0)

    def p_cover(self, home: str, away: str, line: float, neutral: bool = False) -> float:
        """P(home margin > line)."""
        return float(1 - norm.cdf((line - self.margin(home, away, neutral)) / self.sigma))

    def p_over(self, home: str, away: str, line: float, sigma_total: float = 16.0) -> float:
        return float(1 - norm.cdf((line - self.total(home, away)) / sigma_total))


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """argmin sum w (y - X b)^2 + sum lam (b - prior)^2."""
    Xw = X * w[:, None]
    A = X.T @ Xw + np.diag(lam)
    b = Xw.T @ y + lam * prior
    return np.linalg.solve(A, b)


def fit_ratings(
    games: pd.DataFrame,
    asof: pd.Timestamp | None = None,
    prior: RatingModel | None = None,
    # Defaults from the 2024-25 walk-forward grid AND alignment with Kalshi's own prices
    # (2026-11-02): no in-season time decay, 35-point blowout cap, light shrinkage. With
    # tau=60/cap=25/unk=3 the ratings were compressed ~2x against the market (logit slope
    # 0.47); these give slope ~0.9, MAE 9.9, Brier 0.167.
    prior_shrink: float = 0.8,
    prior_weight: float = 3.0,
    unknown_weight: float = 2.0,
    tau_days: float = 400.0,
    cap: float = 35.0,
    non_d1: set[str] | None = None,
    non_d1_prior: float = -28.0,
) -> RatingModel:
    """Fit on completed games with date < asof (strictly: no look-ahead).

    `non_d1`: team ids outside Division I; their prior rating is `non_d1_prior` (points
    below an average D-I team) instead of 0, with the unknown-team weight.
    """
    non_d1 = non_d1 or set()
    g = games[games.completed].copy()
    if asof is not None:
        g = g[g.game_date < asof]
    g = g.dropna(subset=["home_score", "away_score"])
    teams = sorted(set(g.home_id) | set(g.away_id) | (set(prior.rating) if prior else set()))
    idx = {t: i for i, t in enumerate(teams)}
    n, m = len(g), len(teams)
    if n == 0 and prior is None:
        return RatingModel()
    ref = asof if asof is not None else (g.game_date.max() if n else pd.Timestamp.now())
    age = (ref - g.game_date).dt.days.to_numpy(dtype=float) if n else np.zeros(0)
    w = np.exp(-age / tau_days)

    # margin model: columns = team ratings + home advantage
    X = np.zeros((n, m + 1))
    hi = g.home_id.map(idx).to_numpy()
    ai = g.away_id.map(idx).to_numpy()
    X[np.arange(n), hi] = 1.0
    X[np.arange(n), ai] = -1.0
    X[:, m] = (~g.neutral.to_numpy()).astype(float)
    y = np.clip(g.margin.to_numpy(dtype=float), -cap, cap)
    prior_vec = np.zeros(m + 1)
    lam = np.full(m + 1, unknown_weight)
    for t, i in idx.items():
        if t in non_d1:
            prior_vec[i] = non_d1_prior
    if prior is not None:
        for t, i in idx.items():
            if t in prior.rating:
                prior_vec[i] = prior_shrink * prior.rating[t] if t not in non_d1 else prior.rating[t]
                lam[i] = prior_weight
    prior_vec[m], lam[m] = 3.2, 2.0  # home advantage prior ~3.2 points
    lam[:m] += 1e-3  # identifiability (ratings are only defined up to a constant)
    beta = _ridge(X, y, w, lam, prior_vec) if n else prior_vec
    rating = {t: float(beta[i]) for t, i in idx.items()}
    d1_vals = [r for t, r in rating.items() if t not in non_d1] or list(rating.values())
    mean = float(np.mean(d1_vals))  # zero = average Division-I team
    rating = {t: r - mean for t, r in rating.items()}
    hfa = float(beta[m])
    resid = y - X @ beta if n else np.array([11.0])
    sigma = float(np.sqrt(np.average(resid**2, weights=w))) if n > 50 else (prior.sigma if prior else 11.0)

    # totals: points_home = base + off_home + def_away ; points_away = base + off_away + def_home
    off = dfn = {}
    base = prior.base_points if prior else 72.0
    if n > 0:
        Xp = np.zeros((2 * n, 2 * m))
        Xp[np.arange(n), hi] = 1.0
        Xp[np.arange(n), m + ai] = 1.0
        Xp[n + np.arange(n), ai] = 1.0
        Xp[n + np.arange(n), m + hi] = 1.0
        yp = np.concatenate([g.home_score.to_numpy(float), g.away_score.to_numpy(float)])
        base = float(np.average(yp, weights=np.concatenate([w, w])))
        pv = np.zeros(2 * m)
        lp = np.full(2 * m, unknown_weight)
        if prior is not None:
            for t, i in idx.items():
                if t in prior.off:
                    pv[i], pv[m + i] = prior_shrink * prior.off[t], prior_shrink * prior.dfn[t]
                    lp[i] = lp[m + i] = prior_weight
        bp = _ridge(Xp, yp - base, np.concatenate([w, w]), lp, pv)
        off = {t: float(bp[i]) for t, i in idx.items()}
        dfn = {t: float(bp[m + i]) for t, i in idx.items()}
    elif prior is not None:
        off, dfn = {t: prior_shrink * v for t, v in prior.off.items()}, {t: prior_shrink * v for t, v in prior.dfn.items()}
    return RatingModel(rating, off, dfn, hfa, sigma, base, n)
