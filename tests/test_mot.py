"""MOT metrics: perfect tracking scores 1, identity switches are punished, and our
HOTA agrees with the reference TrackEval implementation when it is installed."""
import numpy as np
import pytest

from hoops.eval.mot import clear_mot, hota, read_mot, trackeval_hota, write_mot


def synth(n_frames=50, n_ids=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_ids):
        x0, y0 = rng.uniform(0, 800), rng.uniform(0, 400)
        vx, vy = rng.uniform(-3, 3), rng.uniform(-2, 2)
        for f in range(1, n_frames + 1):
            rows.append([f, i + 1, x0 + vx * f, y0 + vy * f, 40, 100, 1, -1, -1, -1])
    return np.array(rows, float)


def test_perfect_tracking_scores_one():
    gt = synth()
    m = hota(gt, gt)
    assert abs(m["HOTA"] - 1.0) < 1e-9 and abs(m["DetA"] - 1.0) < 1e-9 and abs(m["AssA"] - 1.0) < 1e-9
    c = clear_mot(gt, gt)
    assert c["MOTA"] == 1.0 and abs(c["IDF1"] - 1.0) < 1e-9 and c["IDSW"] == 0


def test_id_switch_reduces_assa_not_deta():
    gt = synth()
    res = gt.copy()
    # swap ids 1 and 2 for the second half of the clip
    half = res[:, 0] > 25
    one, two = half & (res[:, 1] == 1), half & (res[:, 1] == 2)
    res[one, 1], res[two, 1] = 2, 1
    m = hota(gt, res)
    assert abs(m["DetA"] - 1.0) < 1e-9
    assert m["AssA"] < 0.7
    c = clear_mot(gt, res)
    assert c["IDSW"] == 2 and c["IDF1"] < 1.0


def test_missed_detections_reduce_deta():
    gt = synth()
    res = gt[gt[:, 0] % 2 == 1]  # every other frame missing
    m = hota(gt, res)
    assert 0.4 < m["DetA"] < 0.6
    # HOTA's AssA counts a gt track's missed frames as association misses (FNA),
    # so dropping half the frames halves AssA too: this matches TrackEval.
    assert abs(m["AssA"] - 0.5) < 1e-9


def test_write_read_roundtrip(tmp_path):
    gt = synth(n_frames=5)
    rows = [(int(r[0]), int(r[1]), r[2], r[3], r[2] + r[4], r[3] + r[5], 0.9) for r in gt]
    p = tmp_path / "res.txt"
    write_mot(p, rows)
    back = read_mot(p)
    np.testing.assert_allclose(back[:, :6], gt[:, :6], atol=0.01)


def test_matches_trackeval_if_available(tmp_path):
    pytest.importorskip("trackeval")
    gt = synth(n_frames=40, n_ids=4)
    rng = np.random.default_rng(1)
    res = gt.copy()
    res[:, 2:4] += rng.normal(0, 6, (len(res), 2))  # jitter boxes
    res = res[rng.random(len(res)) > 0.1]  # drop 10%
    half = res[:, 0] > 20
    a, b = half & (res[:, 1] == 1), half & (res[:, 1] == 3)
    res[a, 1], res[b, 1] = 3, 1
    gt_f, res_f = tmp_path / "gt.txt", tmp_path / "res.txt"
    np.savetxt(gt_f, gt, delimiter=",", fmt="%.3f")
    np.savetxt(res_f, res, delimiter=",", fmt="%.3f")
    ref = trackeval_hota(gt_f, res_f)
    if ref is None:
        pytest.skip("trackeval import failed")
    ours = hota(gt, res)
    assert abs(ours["HOTA"] - ref["HOTA"]) < 0.01, (ours, ref)
    assert abs(ours["DetA"] - ref["DetA"]) < 0.01
    assert abs(ours["AssA"] - ref["AssA"]) < 0.01
