"""Interacting Multiple Model (IMM) filter.

Runs several Kalman filters with different motion assumptions in parallel and
keeps a probability for each. Each frame the filters are first *mixed* (every
filter is re-initialised from a probability-weighted blend of all filters, so a
newly plausible model does not start cold), then predicted and updated, then
the probabilities are revised from how well each model explained the measurement.

Default bank: constant velocity, constant acceleration, and left/right
coordinated turns. This is the mechanism ARC-SORT credits for its SportsMOT gain.
"""
from __future__ import annotations

import numpy as np

from .base import MotionModel
from .kalman import LinearKalman


class IMM(MotionModel):
    def __init__(
        self,
        z0: np.ndarray,
        n_pos: int | None = None,
        modes: tuple[tuple[str, float], ...] = (
            ("cv", 0.0),
            ("ca", 0.0),
            ("ct", 0.15),
            ("ct", -0.15),
        ),
        p_stay: float = 0.95,
        **kalman_kw,
    ):
        z0 = np.asarray(z0, dtype=float)
        self.models: list[LinearKalman] = [
            LinearKalman(z0, n_pos=n_pos, mode=m, omega=w, **kalman_kw) for m, w in modes
        ]
        self.measurement_dim = self.models[0].measurement_dim
        M = len(self.models)
        self.mu = np.full(M, 1.0 / M)
        off = (1.0 - p_stay) / max(M - 1, 1)
        self.Pi = np.full((M, M), off)
        np.fill_diagonal(self.Pi, p_stay)
        self.c_pred = self.mu.copy()
        self.age = 0

    # -- IMM steps -------------------------------------------------------
    def _mix(self) -> None:
        M = len(self.models)
        c = self.Pi.T @ self.mu  # predicted mode probabilities
        c = np.maximum(c, 1e-12)
        w = (self.Pi * self.mu[:, None]) / c[None, :]  # w[i, j] = P(i at t-1 | j at t)
        xs = [m.x.copy() for m in self.models]
        Ps = [m.P.copy() for m in self.models]
        for j, mj in enumerate(self.models):
            x0 = sum(w[i, j] * xs[i] for i in range(M))
            P0 = np.zeros_like(Ps[0])
            for i in range(M):
                d = xs[i] - x0
                P0 += w[i, j] * (Ps[i] + np.outer(d, d))
            mj.x, mj.P = x0, P0
        self.c_pred = c

    def predict(self, dt: float = 1.0) -> np.ndarray:
        self._mix()
        for m in self.models:
            m.predict(dt)
        self.age += 1
        return self.mean

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        L = np.array([m.likelihood(z) for m in self.models])
        for m in self.models:
            m.update(z)
        mu = L * self.c_pred
        if mu.sum() <= 0 or not np.isfinite(mu.sum()):
            mu = self.c_pred
        self.mu = mu / mu.sum()
        self.age = 0

    # -- combined estimate ----------------------------------------------
    @property
    def x(self) -> np.ndarray:
        return sum(mu * m.x for mu, m in zip(self.mu, self.models))

    @property
    def P(self) -> np.ndarray:
        x = self.x
        P = np.zeros_like(self.models[0].P)
        for mu, m in zip(self.mu, self.models):
            d = m.x - x
            P += mu * (m.P + np.outer(d, d))
        return P

    @property
    def mean(self) -> np.ndarray:
        H = self.models[0].H
        return H @ self.x

    @property
    def cov(self) -> np.ndarray:
        H = self.models[0].H
        return H @ self.P @ H.T + self.models[0]._R()

    def apply_affine(self, a: np.ndarray, b: np.ndarray) -> None:
        for m in self.models:
            m.apply_affine(a, b)

    @property
    def mode_probs(self) -> np.ndarray:
        return self.mu.copy()
