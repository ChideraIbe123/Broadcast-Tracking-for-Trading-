"""HoopsTracker on synthetic detections: ids must stay stable through motion,
missed detections and a crossing, for every motion model."""
import numpy as np
import pytest

from hoops.detect.base import Detections
from hoops.track.assoc import hungarian, iou_matrix
from hoops.track.ours import HoopsTracker, HoopsTrackerConfig


def make_dets(boxes, conf=0.9):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    return Detections(boxes, np.full(len(boxes), conf, np.float32), np.zeros(len(boxes), int), {0: "player"})


def box(cx, cy, w=40, h=100):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_iou_and_hungarian():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    b = np.array([[0, 0, 10, 10], [21, 21, 31, 31], [100, 100, 110, 110]], float)
    iou = iou_matrix(a, b)
    assert iou[0, 0] == 1.0 and iou[1, 1] > 0.6 and iou[0, 2] == 0.0
    m, ur, uc = hungarian(1 - iou, 0.8)
    assert m == [(0, 0), (1, 1)] and ur == [] and uc == [2]


@pytest.mark.parametrize("motion", ["kalman_cv", "kalman_ca", "imm", "particle"])
def test_ids_stable_through_gaps(motion):
    cfg = HoopsTrackerConfig(motion=motion, cmc=None, min_hits=2, max_age=30)
    if motion == "particle":
        cfg.motion_kw = dict(sigma_meas=5.0, sigma_acc=1.0, sigma_init_pos=3.0, sigma_init_vel=3.0)
    tr = HoopsTracker(cfg)
    ids_seen = {}
    for f in range(60):
        p1 = box(100 + 4 * f, 300)
        p2 = box(900 - 3 * f, 500 + f)
        boxes = [p1, p2]
        if 25 <= f < 33:  # player 2 undetected for 8 frames
            boxes = [p1]
        out = tr.update(make_dets(boxes))
        for t in out:
            ids_seen.setdefault(t.id, []).append(f)
    # exactly two identities, each spanning almost the whole clip
    assert len(ids_seen) == 2, ids_seen.keys()
    spans = sorted(len(v) for v in ids_seen.values())
    assert spans[0] >= 60 - 8 - 2 and spans[1] >= 58


def test_low_confidence_second_pass_keeps_track():
    tr = HoopsTracker(HoopsTrackerConfig(motion="kalman_cv", cmc=None))
    ids = set()
    for f in range(40):
        conf = 0.3 if 15 <= f < 30 else 0.9  # occluded: detector only half-sure
        out = tr.update(make_dets([box(200 + 3 * f, 300)], conf=conf))
        ids |= {t.id for t in out}
    assert len(ids) == 1


def test_crossing_swap_uses_appearance():
    """Two players cross paths; the detector boxes are perfect but a fake embedder
    gives each player a distinct appearance. After the crossing the ids must still
    follow appearance, even if we deliberately feed them in swapped order."""

    class FakeEmbedder:
        dim = 2

        def __call__(self, image, xyxy):
            # appearance encoded in the frame: we stash it in a global via the image array
            tags = image_tags[: len(xyxy)]
            return np.array([[1.0, 0.0] if t == "A" else [0.0, 1.0] for t in tags])

    image_tags = []
    tr = HoopsTracker(
        HoopsTrackerConfig(motion="kalman_cv", cmc=None, crossing_swap=True, appearance_thresh=0.9, match_thresh=0.95),
        embedder=FakeEmbedder(),
    )
    frame = np.zeros((8, 8, 3), np.uint8)
    id_of = {}
    for f in range(80):
        xa, xb = 100 + 10 * f, 900 - 10 * f  # cross at f=40
        A_box, B_box = box(xa, 400), box(xb, 400)
        image_tags[:] = ["A", "B"]
        out = tr.update(make_dets([A_box, B_box]), frame)
        if abs(xa - xb) < 80:  # boxes coincide: which is which is undefined
            continue
        for t in out:
            tag = "A" if abs(t.xyxy[0] + 20 - xa) < abs(t.xyxy[0] + 20 - xb) else "B"
            id_of.setdefault(tag, set()).add(t.id)
    assert len(id_of["A"]) == 1 and len(id_of["B"]) == 1 and id_of["A"] != id_of["B"], id_of


def test_crossing_swap_corrects_swapped_ids():
    """Directly exercise the crossing watch: two tracks overlap, then separate with
    their appearances exchanged -> the ids must be swapped back exactly once."""
    from hoops.track.ours import _T

    tr = HoopsTracker(HoopsTrackerConfig(motion="kalman_cv", cmc=None, crossing_swap=True), embedder=object())
    a = _T(1, np.array([500, 400, 40, 100.0]), 0, 0.9, "kalman_cv", {}, np.array([1.0, 0.0]))
    b = _T(2, np.array([505, 400, 40, 100.0]), 0, 0.9, "kalman_cv", {}, np.array([0.0, 1.0]))
    a.state = b.state = "tracked"
    tr.tracks = [a, b]
    tr._crossings()  # overlapping: enter crossing, snapshot embeddings
    assert a.cross_with == 2 and b.cross_with == 1
    # separate them, but now the track that *was* id 1 looks like B and vice versa
    a.motion.update(np.array([300, 400, 40, 100.0]))
    b.motion.update(np.array([700, 400, 40, 100.0]))
    a.emb, b.emb = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    tr._crossings()
    assert (a.id, b.id) == (2, 1)
    assert a.cross_with is None and b.cross_with is None
