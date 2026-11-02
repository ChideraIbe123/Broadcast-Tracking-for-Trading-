"""Bootstrap particle filter with constant-velocity dynamics and random acceleration.

Keeps N hypotheses instead of one Gaussian, so it can represent "the ball is
behind this player OR that one" and survive long stretches without a
measurement. Costs ~N times a Kalman filter; intended for the ball (and for
benchmarking against Kalman/IMM on players).
"""
from __future__ import annotations

import numpy as np

from .base import MotionModel


class ParticleFilter(MotionModel):
    def __init__(
        self,
        z0: np.ndarray,
        n_particles: int = 500,
        sigma_init_pos: float = 1.0,
        sigma_init_vel: float = 1.0,
        sigma_acc: float = 0.5,
        sigma_meas: float = 1.0,
        resample_threshold: float = 0.5,
        seed: int | None = 0,
    ):
        z0 = np.asarray(z0, dtype=float)
        self.n = len(z0)
        self.measurement_dim = self.n
        self.N = n_particles
        self.rng = np.random.default_rng(seed)
        self.sigma_acc, self.sigma_meas = sigma_acc, sigma_meas
        self.resample_threshold = resample_threshold
        self.pos = z0 + self.rng.normal(0, sigma_init_pos, (self.N, self.n))
        self.vel = self.rng.normal(0, sigma_init_vel, (self.N, self.n))
        self.w = np.full(self.N, 1.0 / self.N)
        self.age = 0

    def predict(self, dt: float = 1.0) -> np.ndarray:
        acc = self.rng.normal(0, self.sigma_acc, (self.N, self.n))
        self.pos += self.vel * dt + 0.5 * acc * dt * dt
        self.vel += acc * dt
        self.age += 1
        return self.mean

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=float)
        d2 = ((self.pos - z) ** 2).sum(axis=1)
        logw = -0.5 * d2 / (self.sigma_meas**2)
        logw -= logw.max()
        w = self.w * np.exp(logw)
        if w.sum() <= 0 or not np.isfinite(w.sum()):
            w = np.full(self.N, 1.0 / self.N)
        self.w = w / w.sum()
        n_eff = 1.0 / (self.w**2).sum()
        if n_eff < self.resample_threshold * self.N:
            self._resample()
        self.age = 0

    def _resample(self) -> None:
        """Systematic resampling: low variance, O(N)."""
        positions = (self.rng.random() + np.arange(self.N)) / self.N
        cumsum = np.cumsum(self.w)
        cumsum[-1] = 1.0
        idx = np.searchsorted(cumsum, positions)
        self.pos, self.vel = self.pos[idx].copy(), self.vel[idx].copy()
        self.w = np.full(self.N, 1.0 / self.N)

    @property
    def mean(self) -> np.ndarray:
        return self.w @ self.pos

    @property
    def cov(self) -> np.ndarray:
        d = self.pos - self.mean
        return (self.w[:, None] * d).T @ d + (self.sigma_meas**2) * np.eye(self.n)

    def apply_affine(self, a: np.ndarray, b: np.ndarray) -> None:
        self.pos[:, :2] = self.pos[:, :2] @ a.T + b
        self.vel[:, :2] = self.vel[:, :2] @ a.T
