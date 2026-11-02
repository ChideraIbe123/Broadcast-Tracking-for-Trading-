"""Probabilistic evaluation for win-probability models."""
from __future__ import annotations

import numpy as np


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bin centres, observed frequency, count) for a calibration curve."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    obs = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)])
    cnt = np.array([(idx == b).sum() for b in range(n_bins)])
    pred = np.array([p[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)])
    return pred, obs, cnt


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    pred, obs, cnt = reliability(y, p, n_bins)
    m = cnt > 0
    return float(np.sum(cnt[m] * np.abs(pred[m] - obs[m])) / cnt.sum())


def evaluate(y: np.ndarray, p: np.ndarray, secs_left: np.ndarray | None = None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"log_loss": log_loss(y, p), "brier": brier(y, p), "ece": ece(y, p), "n": int(len(y))}
    if secs_left is not None:
        # how the model does at different stages of the game
        for name, lo, hi in (("q1", 2160, 2880), ("q2", 1440, 2160), ("q3", 720, 1440), ("q4", 0, 720), ("last2min", 0, 120)):
            m = (secs_left >= lo) & (secs_left < hi)
            if m.sum() > 100:
                out[f"log_loss_{name}"] = log_loss(y[m], p[m])
    return out
