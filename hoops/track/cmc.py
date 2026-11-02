"""Camera-motion compensation (CMC).

Broadcast cameras pan and zoom, so every track prediction is wrong by the same
global shift each frame. We estimate a 2D similarity/affine transform between
consecutive frames from sparse optical flow on *background* features (detections
are masked out so players do not pollute the estimate) and apply it to all tracks
before association. Same idea as BoT-SORT's GMC; ECC is the slower, more accurate
alternative.
"""
from __future__ import annotations

import cv2
import numpy as np


class SparseFlowCMC:
    def __init__(self, downscale: int = 2, max_corners: int = 1000, min_inliers: int = 20):
        self.downscale = downscale
        self.max_corners = max_corners
        self.min_inliers = min_inliers
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None

    def _prep(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            gray = cv2.resize(gray, None, fx=1 / self.downscale, fy=1 / self.downscale, interpolation=cv2.INTER_AREA)
        mask = np.full(gray.shape, 255, np.uint8)
        # ignore the frame border (broadcast graphics) and every detection box
        h, w = gray.shape
        mask[: int(0.03 * h), :] = 0
        mask[int(0.97 * h) :, :] = 0
        if boxes_xyxy is not None and len(boxes_xyxy):
            for x1, y1, x2, y2 in np.asarray(boxes_xyxy) / self.downscale:
                mask[max(int(y1), 0) : int(y2) + 1, max(int(x1), 0) : int(x2) + 1] = 0
        return gray, mask

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (A, b) with x_curr = A @ x_prev + b in full-resolution pixels."""
        gray, mask = self._prep(frame_bgr, boxes_xyxy)
        A, b = np.eye(2), np.zeros(2)
        pts = cv2.goodFeaturesToTrack(gray, self.max_corners, 0.01, 7, mask=mask, blockSize=3)
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= self.min_inliers:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
            ok = status.ravel() == 1
            if ok.sum() >= self.min_inliers:
                M, inl = cv2.estimateAffinePartial2D(self.prev_pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is not None and inl is not None and inl.sum() >= self.min_inliers:
                    A = M[:, :2]
                    b = M[:, 2] * self.downscale
        self.prev_gray = gray
        self.prev_pts = pts
        return A, b


class NoCMC:
    def reset(self) -> None:
        return None

    def __call__(self, frame_bgr, boxes_xyxy=None):
        return np.eye(2), np.zeros(2)


def make_cmc(method: str | None):
    if method in (None, "none", False):
        return NoCMC()
    if method == "sparseOptFlow":
        return SparseFlowCMC()
    raise ValueError(f"unknown CMC method {method}")
