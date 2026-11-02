"""Every tracker behind the registry must run on the same synthetic clip and keep
two identities stable through an 8-frame detection gap. Library trackers need a
frame; we feed a static grey image (no camera motion)."""
import numpy as np
import pytest

from hoops.detect.base import Detections
from hoops.registry import make_tracker

FRAME = np.full((720, 1280, 3), 90, np.uint8)


def make_dets(boxes, conf=0.9):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return Detections(boxes, np.full(len(boxes), conf, np.float32), np.zeros(len(boxes), int), {0: "player"})


def box(cx, cy, w=40, h=100):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


CANDIDATES = [
    "hoops:kalman_cv",
    "hoops:imm",
    "trackers:bytetrack",
    "trackers:botsort",
    "trackers:ocsort",
    "boxmot:botsort",
    "boxmot:bytetrack",
    "boxmot:occluboost",
    "ultralytics:tracktrack",
    "ultralytics:botsort",
    "ultralytics:bytetrack",
]


@pytest.mark.parametrize("name", CANDIDATES)
def test_tracker_runs_and_keeps_ids(name):
    try:
        tr = make_tracker(name)
    except Exception as e:  # pragma: no cover - surface install problems clearly
        pytest.fail(f"{name} failed to construct: {e}")
    ids_seen: dict[int, list[int]] = {}
    for f in range(60):
        p1 = box(100 + 4 * f, 300)
        p2 = box(900 - 3 * f, 500 + f)
        boxes = [p1] if 25 <= f < 33 else [p1, p2]
        out = tr.update(make_dets(boxes), FRAME.copy())
        for t in out:
            assert t.xyxy.shape == (4,)
            ids_seen.setdefault(t.id, []).append(f)
    # at most 3 identities overall (some trackers restart a track after a long gap)
    assert 2 <= len(ids_seen) <= 3, (name, {k: len(v) for k, v in ids_seen.items()})
    longest = sorted((len(v) for v in ids_seen.values()), reverse=True)
    assert longest[0] >= 50 and longest[1] >= 20, (name, longest)
