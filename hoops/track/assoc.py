"""Association utilities: IoU variants, appearance cost and Hungarian matching."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def expand_boxes(xyxy: np.ndarray, scale: float) -> np.ndarray:
    """Scale boxes about their centre (Deep-EIoU's ExpansionIoU uses scale > 1 so that
    fast-moving players whose boxes barely overlap between frames still associate)."""
    if scale == 1.0:
        return xyxy
    b = np.asarray(xyxy, dtype=float)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    w, h = (b[:, 2] - b[:, 0]) * scale, (b[:, 3] - b[:, 1]) * scale
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def iou_matrix(a: np.ndarray, b: np.ndarray, expand: float = 1.0) -> np.ndarray:
    """(len(a), len(b)) IoU; `expand` > 1 gives ExpansionIoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = expand_boxes(a, expand), expand_boxes(b, expand)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def cosine_cost(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    """1 - cosine similarity between L2-normalised embeddings; None when unavailable."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return None
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return 1.0 - a @ b.T


def fuse_costs(
    iou_cost: np.ndarray,
    emb_cost: np.ndarray | None,
    proximity_thresh: float = 0.5,
    appearance_thresh: float = 0.25,
) -> np.ndarray:
    """BoT-SORT fusion: appearance is only trusted where boxes are reasonably close,
    and only when it is confident; the final cost is the smaller of the two."""
    if emb_cost is None:
        return iou_cost
    emb = emb_cost.copy()
    emb[emb > appearance_thresh] = 1.0
    emb[iou_cost > proximity_thresh] = 1.0
    return np.minimum(iou_cost, emb)


def hungarian(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Min-cost assignment with rejection of pairs whose cost exceeds `thresh`.

    Returns (matches, unmatched_rows, unmatched_cols).
    """
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    c = np.where(np.isfinite(cost), cost, 1e6)
    rows, cols = linear_sum_assignment(c)
    matches, mr, mc = [], set(range(cost.shape[0])), set(range(cost.shape[1]))
    for r, c_ in zip(rows, cols):
        if cost[r, c_] <= thresh:
            matches.append((int(r), int(c_)))
            mr.discard(int(r))
            mc.discard(int(c_))
    return matches, sorted(mr), sorted(mc)
