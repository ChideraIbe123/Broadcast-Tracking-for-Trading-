"""Common interface for motion models used by the tracker and the ball filter.

A motion model keeps a belief about where an object is and how it moves, and
answers two questions each frame:
  predict(dt): where will it be next, and how unsure am I?
  update(z):   given a new measurement z, revise the belief.

Measurement convention (image or court plane): z = [x, y, w, h] for boxes,
or z = [x, y] for points. Implementations declare `measurement_dim`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MotionModel(ABC):
    measurement_dim: int

    @abstractmethod
    def predict(self, dt: float = 1.0) -> np.ndarray:
        """Advance the belief by dt frames; return predicted measurement."""

    @abstractmethod
    def update(self, z: np.ndarray) -> None:
        """Incorporate measurement z."""

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """Current expected measurement (same shape as z)."""

    @property
    @abstractmethod
    def cov(self) -> np.ndarray:
        """Covariance of the current expected measurement."""

    def gate_distance(self, z: np.ndarray) -> float:
        """Squared Mahalanobis distance from the predicted measurement to z.

        Used to decide whether a detection is even a plausible match. A value above
        the chi-square threshold for `measurement_dim` degrees of freedom is rejected.
        """
        d = np.asarray(z, dtype=float) - self.mean
        s = self.cov + 1e-9 * np.eye(len(d))
        return float(d @ np.linalg.solve(s, d))

    def apply_affine(self, a: np.ndarray, b: np.ndarray) -> None:  # noqa: B027
        """Camera-motion compensation hook: x' = A x + b on the position part.

        Default is a no-op; Kalman models override it.
        """
        return None


# chi-square 0.95 quantiles by degrees of freedom, used for gating
CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592}
