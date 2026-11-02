from .base import CHI2_95, MotionModel
from .imm import IMM
from .kalman import KalmanCA, KalmanCT, KalmanCV, LinearKalman
from .particle import ParticleFilter

REGISTRY = {
    "kalman_cv": KalmanCV,
    "kalman_ca": KalmanCA,
    "imm": IMM,
    "particle": ParticleFilter,
}


def make_motion(name: str, z0, **kw) -> MotionModel:
    """Factory used by the tracker and the benchmark harness."""
    return REGISTRY[name](z0, **kw)


__all__ = [
    "CHI2_95",
    "MotionModel",
    "IMM",
    "KalmanCA",
    "KalmanCT",
    "KalmanCV",
    "LinearKalman",
    "ParticleFilter",
    "REGISTRY",
    "make_motion",
]
