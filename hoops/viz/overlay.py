"""Draw tracks, court lines and gate status onto frames."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

from hoops.court.geometry import court_lines
from hoops.court.homography import court_to_image
from hoops.track.base import Track


def id_color(i: int) -> tuple[int, int, int]:
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(img: np.ndarray, tracks: list[Track], team: dict[int, int] | None = None) -> np.ndarray:
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.xyxy)
        color = id_color(t.id)
        if team is not None and t.id in team:
            color = [(60, 60, 230), (230, 160, 40), (200, 200, 200)][team[t.id] % 3]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{t.id}"
        cv2.putText(img, label, (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def draw_court(img: np.ndarray, H: np.ndarray, color=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Project the court model through H and draw it. If the lines sit on the paint, H is right."""
    h, w = img.shape[:2]
    for pts in court_lines().values():
        px = court_to_image(H, pts)
        ok = np.isfinite(px).all(axis=1) & (np.abs(px[:, 0]) < 4 * w) & (np.abs(px[:, 1]) < 4 * h)
        if ok.sum() < 2:
            continue
        poly = px[ok].astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], False, color, thickness, cv2.LINE_AA)
    return img


def draw_status(img: np.ndarray, text: str, ok: bool = True) -> np.ndarray:
    color = (40, 200, 40) if ok else (30, 30, 230)
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img
