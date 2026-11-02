"""Linear Kalman filters on a shared state layout [pos, vel, acc] (3 * n_pos).

Sharing one state layout across constant-velocity (CV), constant-acceleration (CA)
and coordinated-turn (CT) models lets the IMM filter mix them without padding.

n_pos = 4 for boxes (cx, cy, w, h) in pixels; n_pos = 2 for points on the court.
"""
from __future__ import annotations

import numpy as np

from .base import MotionModel


def _block_F(n: int, dt: float, mode: str, omega: float = 0.0) -> np.ndarray:
    """State transition for [pos, vel, acc] under the given motion mode."""
    I = np.eye(n)
    Z = np.zeros((n, n))
    if mode == "cv":
        # pos += vel*dt ; vel = vel ; acc -> 0
        F = np.block([[I, dt * I, Z], [Z, I, Z], [Z, Z, Z]])
    elif mode == "ca":
        F = np.block([[I, dt * I, 0.5 * dt * dt * I], [Z, I, dt * I], [Z, Z, I]])
    elif mode == "ct":
        # coordinated turn on the first two position dims (x, y); CV elsewhere
        F = np.block([[I, dt * I, Z], [Z, I, Z], [Z, Z, Z]])
        if abs(omega) > 1e-9:
            s, c = np.sin(omega * dt), np.cos(omega * dt)
            F[0, n + 0] = s / omega
            F[0, n + 1] = -(1 - c) / omega
            F[1, n + 0] = (1 - c) / omega
            F[1, n + 1] = s / omega
            F[n + 0, n + 0] = c
            F[n + 0, n + 1] = -s
            F[n + 1, n + 0] = s
            F[n + 1, n + 1] = c
    else:
        raise ValueError(mode)
    return F


class LinearKalman(MotionModel):
    """Generic linear Kalman filter on [pos, vel, acc].

    q_pos/q_vel/q_acc: process-noise std per frame for each block.
    r_meas: measurement-noise std. If `scale_by_size` and n_pos == 4, all stds are
    multiplied by the current box height (the SORT family convention), so the same
    settings work for near and far players.
    """

    def __init__(
        self,
        z0: np.ndarray,
        n_pos: int | None = None,
        mode: str = "cv",
        omega: float = 0.0,
        q_pos: float = 1 / 20,
        q_vel: float = 1 / 160,
        q_acc: float = 1 / 500,
        r_meas: float = 1 / 20,
        p0_vel: float = 10.0,
        scale_by_size: bool = True,
    ):
        z0 = np.asarray(z0, dtype=float)
        self.n = n_pos or len(z0)
        self.measurement_dim = self.n
        self.mode, self.omega = mode, omega
        self.q = (q_pos, q_vel, q_acc)
        self.r = r_meas
        self.scale_by_size = scale_by_size and self.n == 4
        self.x = np.zeros(3 * self.n)
        self.x[: self.n] = z0
        s = self._scale()
        self.P = np.diag(
            np.concatenate(
                [
                    (2 * r_meas * s) ** 2 * np.ones(self.n),
                    (p0_vel * r_meas * s) ** 2 * np.ones(self.n),
                    (p0_vel * r_meas * s) ** 2 * np.ones(self.n),
                ]
            )
        )
        self.H = np.hstack([np.eye(self.n), np.zeros((self.n, 2 * self.n))])
        self.age = 0

    # -- helpers ---------------------------------------------------------
    def _scale(self) -> float:
        return float(max(self.x[3], 1.0)) if self.scale_by_size else 1.0

    def _Q(self, dt: float) -> np.ndarray:
        s = self._scale()
        qp, qv, qa = self.q
        d = np.concatenate(
            [
                (qp * s * dt) ** 2 * np.ones(self.n),
                (qv * s * dt) ** 2 * np.ones(self.n),
                (qa * s * dt) ** 2 * np.ones(self.n),
            ]
        )
        if self.mode == "cv":
            d[2 * self.n :] = 0.0
        return np.diag(d)

    def _R(self) -> np.ndarray:
        s = self._scale()
        return (self.r * s) ** 2 * np.eye(self.n)

    # -- MotionModel API -------------------------------------------------
    def predict(self, dt: float = 1.0) -> np.ndarray:
        F = _block_F(self.n, dt, self.mode, self.omega)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self.age += 1
        return self.mean

    def update(self, z: np.ndarray, R: np.ndarray | None = None) -> None:
        """Incorporate z; `R` overrides the measurement covariance for this update
        (used when measurements of different quality feed one filter)."""
        z = np.asarray(z, dtype=float)
        R = self._R() if R is None else np.asarray(R, dtype=float)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ self.H
        # Joseph form keeps P symmetric positive-definite
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.age = 0

    def likelihood(self, z: np.ndarray) -> float:
        """Gaussian likelihood of z under the predicted measurement (for IMM)."""
        z = np.asarray(z, dtype=float)
        S = self.H @ self.P @ self.H.T + self._R()
        y = z - self.H @ self.x
        k = len(y)
        det = max(np.linalg.det(S), 1e-300)
        m = float(y @ np.linalg.solve(S, y))
        return float(np.exp(-0.5 * m) / np.sqrt((2 * np.pi) ** k * det))

    @property
    def mean(self) -> np.ndarray:
        return self.H @ self.x

    @property
    def cov(self) -> np.ndarray:
        return self.H @ self.P @ self.H.T + self._R()

    def apply_affine(self, a: np.ndarray, b: np.ndarray) -> None:
        """Camera-motion compensation: rotate/scale/translate position (x, y),
        rotate/scale velocity and acceleration, and transform P accordingly."""
        n = self.n
        T = np.eye(3 * n)
        T[0:2, 0:2] = a
        T[n : n + 2, n : n + 2] = a
        T[2 * n : 2 * n + 2, 2 * n : 2 * n + 2] = a
        self.x = T @ self.x
        self.x[0:2] += b
        self.P = T @ self.P @ T.T


def KalmanCV(z0, **kw) -> LinearKalman:
    return LinearKalman(z0, mode="cv", **kw)


def KalmanCA(z0, **kw) -> LinearKalman:
    return LinearKalman(z0, mode="ca", **kw)


def KalmanCT(z0, omega: float, **kw) -> LinearKalman:
    return LinearKalman(z0, mode="ct", omega=omega, **kw)
