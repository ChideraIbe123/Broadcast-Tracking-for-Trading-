"""Top-down court rendering ("minimap") with player dots."""
from __future__ import annotations

import cv2
import numpy as np

from hoops.court.geometry import court_lines, cur

from .overlay import id_color


class Minimap:
    def __init__(self, px_per_m: float = 20.0, margin_m: float = 1.0):
        c = cur()
        self.L, self.W = c.length, c.width
        self.s = px_per_m
        self.m = margin_m
        self.w = int((self.L + 2 * margin_m) * px_per_m)
        self.h = int((self.W + 2 * margin_m) * px_per_m)
        self.base = self._draw_base()

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        u = (xy[:, 0] + self.L / 2 + self.m) * self.s
        v = (self.W / 2 - xy[:, 1] + self.m) * self.s  # +y court = up on screen
        return np.stack([u, v], 1)

    def _draw_base(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), (160, 120, 80), np.uint8)  # wood-ish
        for pts in court_lines().values():
            poly = self.to_px(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], False, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def render(self, xy: np.ndarray, ids: list[int], team: dict[int, int] | None = None, ball_xy=None) -> np.ndarray:
        img = self.base.copy()
        for p, i in zip(self.to_px(xy), ids):
            color = id_color(i)
            if team is not None and i in team:
                color = [(60, 60, 230), (230, 160, 40), (200, 200, 200)][team[i] % 3]
            cv2.circle(img, (int(p[0]), int(p[1])), 7, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (int(p[0]) + 8, int(p[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        if ball_xy is not None:
            b = self.to_px(np.asarray(ball_xy))[0]
            cv2.circle(img, (int(b[0]), int(b[1])), 5, (0, 140, 255), -1, cv2.LINE_AA)
        return img

    def attach(self, frame: np.ndarray, mini: np.ndarray, corner: str = "br") -> np.ndarray:
        """Paste the minimap into a corner of the broadcast frame."""
        H, W = frame.shape[:2]
        scale = min(0.35 * W / mini.shape[1], 0.35 * H / mini.shape[0])
        m = cv2.resize(mini, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = m.shape[:2]
        y0 = H - h - 10 if "b" in corner else 36
        x0 = W - w - 10 if "r" in corner else 10
        frame[y0 : y0 + h, x0 : x0 + w] = m
        return frame
