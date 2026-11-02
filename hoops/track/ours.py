"""HoopsTracker: our own tracker built from swappable parts so the benchmark can
isolate the effect of each one.

Per frame:
  1. Camera-motion compensation: shift every track by the estimated camera move.
  2. Predict: each track's motion model (Kalman CV/CA, IMM or particle) predicts.
  3. First association: tracked + lost tracks vs high-confidence detections.
     Cost = min(IoU cost, appearance cost) with Mahalanobis gating from the motion
     model's own uncertainty (an uncertain track may match further away).
  4. Second association (ByteTrack): still-unmatched tracks vs low-confidence
     detections on IoU alone, so occluded players are not dropped.
  5. Lifecycle: new tracks start tentative and need `min_hits`; unmatched tracks
     go lost and die after `max_age` frames.
  6. Crossing watch (ARC-SORT style): when two tracks overlap heavily we snapshot
     their appearance; once they separate we check whether the identities came out
     swapped and correct it at most once per crossing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hoops.detect.base import Detections, xywh_to_xyxy
from hoops.motion import CHI2_95, make_motion
from hoops.reid.base import Embedder

from .assoc import cosine_cost, fuse_costs, hungarian, iou_matrix
from .base import Track
from .cmc import make_cmc


@dataclass
class HoopsTrackerConfig:
    motion: str = "kalman_cv"
    motion_kw: dict = field(default_factory=dict)
    cmc: str | None = "sparseOptFlow"
    # Defaults from the SportsMOT basketball val ablation (15 seqs, COCO RF-DETR dets,
    # scripts/ablate_tracker.py, 2026-10-29): gate off (+0.02 HOTA vs gated), thresholds
    # 0.5/0.6, match 0.9 (+0.004), ExpansionIoU 1.2 (+0.003), second match 0.6 (+0.003),
    # CMC on (+0.009). Combined 0.486 HOTA vs 0.464 for ByteTrack on the same detections.
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.9  # max fused cost in the first association
    second_match_thresh: float = 0.6  # max IoU cost in the second association
    iou_expand: float = 1.2  # >1 => ExpansionIoU (Deep-EIoU)
    use_gate: bool = False  # Mahalanobis gate on (x, y) only; off by default (hurt on SportsMOT)
    gate_multiplier: float = 1.0  # scale the chi-square gate
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    emb_momentum: float = 0.9
    max_age: int = 30
    min_hits: int = 2
    crossing_swap: bool = True
    crossing_iou: float = 0.7
    separated_iou: float = 0.2


class _T:
    """Internal track state."""

    __slots__ = ("id", "motion", "cls", "conf", "state", "age", "hits", "emb", "cross_emb", "cross_with", "swapped")

    def __init__(self, id_: int, det_xywh: np.ndarray, cls: int, conf: float, motion: str, motion_kw: dict, emb):
        self.id = id_
        self.motion = make_motion(motion, det_xywh, **motion_kw)
        self.cls, self.conf = int(cls), float(conf)
        self.state = "tentative"
        self.age, self.hits = 0, 1
        self.emb = None if emb is None else emb / max(np.linalg.norm(emb), 1e-9)
        self.cross_emb: np.ndarray | None = None
        self.cross_with: int | None = None
        self.swapped = False

    @property
    def xyxy(self) -> np.ndarray:
        return xywh_to_xyxy(self.motion.mean[None])[0]

    def update_emb(self, emb, momentum: float):
        if emb is None:
            return
        emb = emb / max(np.linalg.norm(emb), 1e-9)
        self.emb = emb if self.emb is None else momentum * self.emb + (1 - momentum) * emb
        self.emb /= max(np.linalg.norm(self.emb), 1e-9)

    def to_track(self) -> Track:
        return Track(self.id, self.xyxy, self.conf, self.cls, self.state, self.age, self.hits, self.emb)


class HoopsTracker:
    name = "hoops"

    def __init__(self, cfg: HoopsTrackerConfig | None = None, embedder: Embedder | None = None):
        self.cfg = cfg or HoopsTrackerConfig()
        self.embedder = embedder
        self.cmc = make_cmc(self.cfg.cmc)
        self.reset()

    def reset(self) -> None:
        self.tracks: list[_T] = []
        self.next_id = 1
        self.frame = 0
        self.cmc.reset()

    # ------------------------------------------------------------------
    def _cost(self, tracks: list[_T], dets: Detections, emb: np.ndarray | None, use_app: bool) -> np.ndarray:
        if not tracks or len(dets) == 0:
            return np.zeros((len(tracks), len(dets)))
        t_xyxy = np.stack([t.xyxy for t in tracks])
        iou_cost = 1.0 - iou_matrix(t_xyxy, dets.xyxy, expand=self.cfg.iou_expand)
        if self.cfg.use_gate:
            # position-only gate: box size changes (w, h) are not a reason to reject a match
            thr = CHI2_95[2] * self.cfg.gate_multiplier
            z = dets.xywh[:, :2]
            for i, t in enumerate(tracks):
                mu, S = t.motion.mean[:2], t.motion.cov[:2, :2] + 1e-9 * np.eye(2)
                d = z - mu
                m = np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)
                iou_cost[i, m > thr] = np.inf
        if use_app and emb is not None and all(t.emb is not None for t in tracks):
            t_emb = np.stack([t.emb for t in tracks])
            return fuse_costs(iou_cost, cosine_cost(t_emb, emb), self.cfg.proximity_thresh, self.cfg.appearance_thresh)
        return iou_cost

    def update(self, detections: Detections, frame: np.ndarray | None = None) -> list[Track]:
        cfg = self.cfg
        self.frame += 1

        # 1. camera motion compensation
        if frame is not None and not isinstance(self.cmc, type(None)):
            A, b = self.cmc(frame, detections.xyxy)
            if not (np.allclose(A, np.eye(2)) and np.allclose(b, 0)):
                for t in self.tracks:
                    t.motion.apply_affine(A, b)

        # 2. predict
        for t in self.tracks:
            t.motion.predict()
            t.age += 1

        # embeddings for all detections (only computed when an embedder is attached)
        emb_all = None
        if self.embedder is not None and frame is not None and len(detections):
            emb_all = self.embedder(frame, detections.xyxy)

        high = detections.conf >= cfg.high_thresh
        low = (detections.conf >= cfg.low_thresh) & ~high
        d_high, d_low = detections.select(high), detections.select(low)
        e_high = None if emb_all is None else emb_all[high]

        # 3. first association: tracked + lost vs high-conf
        active = [t for t in self.tracks if t.state in ("tracked", "lost")]
        cost = self._cost(active, d_high, e_high, use_app=True)
        matches, un_t, un_d = hungarian(cost, cfg.match_thresh)
        for ti, di in matches:
            self._hit(active[ti], d_high, di, None if e_high is None else e_high[di])

        # 4. second association: remaining *tracked* tracks vs low-conf dets, IoU only
        rem = [active[i] for i in un_t if active[i].state == "tracked"]
        cost2 = self._cost(rem, d_low, None, use_app=False)
        m2, un_t2, _ = hungarian(cost2, cfg.second_match_thresh)
        for ti, di in m2:
            self._hit(rem[ti], d_low, di, None)
        matched_ids = {active[i].id for i, _ in matches} | {rem[i].id for i, _ in m2}
        for t in active:
            if t.id not in matched_ids:
                t.state = "lost"

        # tentative tracks: match against the still-unmatched high dets
        tentative = [t for t in self.tracks if t.state == "tentative"]
        d_rest = d_high.select(np.array(un_d, dtype=int)) if un_d else Detections.empty()
        e_rest = None if e_high is None or not un_d else e_high[np.array(un_d)]
        cost3 = self._cost(tentative, d_rest, None, use_app=False)
        m3, un_t3, un_d3 = hungarian(cost3, 0.7)
        for ti, di in m3:
            self._hit(tentative[ti], d_rest, di, None if e_rest is None else e_rest[di])
        for i in un_t3:
            tentative[i].state = "removed"

        # 5. new tracks from leftover high-confidence detections
        for di in un_d3:
            if d_rest.conf[di] >= cfg.new_track_thresh:
                e = None if e_rest is None else e_rest[di]
                self.tracks.append(_T(self.next_id, d_rest.xywh[di], d_rest.cls[di], d_rest.conf[di], cfg.motion, cfg.motion_kw, e))
                self.next_id += 1

        # lifecycle
        self.tracks = [t for t in self.tracks if t.state != "removed" and not (t.state == "lost" and t.age > cfg.max_age)]

        # 6. crossing watch
        if cfg.crossing_swap and self.embedder is not None:
            self._crossings()

        return [t.to_track() for t in self.tracks if t.state == "tracked"]

    # ------------------------------------------------------------------
    def _hit(self, t: _T, dets: Detections, di: int, emb) -> None:
        t.motion.update(dets.xywh[di])
        t.conf = float(dets.conf[di])
        t.cls = int(dets.cls[di])
        t.hits += 1
        t.age = 0
        t.update_emb(emb, self.cfg.emb_momentum)
        if t.state == "tentative" and t.hits >= self.cfg.min_hits:
            t.state = "tracked"
        elif t.state == "lost":
            t.state = "tracked"

    def _crossings(self) -> None:
        tracked = [t for t in self.tracks if t.state == "tracked" and t.emb is not None]
        if len(tracked) < 2:
            return
        boxes = np.stack([t.xyxy for t in tracked])
        iou = iou_matrix(boxes, boxes)
        np.fill_diagonal(iou, 0)
        for i, a in enumerate(tracked):
            for j in range(i + 1, len(tracked)):
                b = tracked[j]
                if iou[i, j] >= self.cfg.crossing_iou and a.cross_with is None and b.cross_with is None:
                    # entering a crossing: freeze pre-crossing appearance
                    a.cross_emb, b.cross_emb = a.emb.copy(), b.emb.copy()
                    a.cross_with, b.cross_with = b.id, a.id
                    a.swapped = b.swapped = False
                elif a.cross_with == b.id and iou[i, j] <= self.cfg.separated_iou:
                    # separated: did identities come out swapped?
                    same = float(a.emb @ a.cross_emb + b.emb @ b.cross_emb)
                    cross = float(a.emb @ b.cross_emb + b.emb @ a.cross_emb)
                    if cross > same + 0.1 and not a.swapped:
                        a.id, b.id = b.id, a.id
                        a.swapped = b.swapped = True
                    a.cross_with = b.cross_with = None
                    a.cross_emb = b.cross_emb = None
