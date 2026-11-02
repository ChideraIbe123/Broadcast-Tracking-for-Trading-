"""MOT-format I/O and metrics (HOTA / IDF1 / MOTA) for the benchmark harness.

SportsMOT ships in MOTChallenge-17 layout:
    <split>/<seq>/img1/000001.jpg, gt/gt.txt, seqinfo.ini
    splits_txt/basketball.txt lists the basketball sequence names.
Result files are written as  <frame>,<id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1  (1-based frames).

Metrics use the reference TrackEval implementation when importable; otherwise
`hota()` falls back to a faithful re-implementation of HOTA(α) averaged over
α ∈ {0.05..0.95}. Both are reported so a discrepancy would be visible.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from hoops.track.assoc import iou_matrix


@dataclass
class Sequence:
    name: str
    root: Path  # .../<split>/<seq>
    n_frames: int
    width: int
    height: int
    fps: float

    @property
    def img_dir(self) -> Path:
        return self.root / "img1"

    @property
    def gt_file(self) -> Path:
        return self.root / "gt" / "gt.txt"

    def frame_path(self, i: int) -> Path:  # 1-based
        return self.img_dir / f"{i:06d}.jpg"


def load_sequence(root: Path) -> Sequence:
    ini = configparser.ConfigParser()
    ini.read(root / "seqinfo.ini")
    s = ini["Sequence"]
    return Sequence(root.name, root, int(s["seqLength"]), int(s["imWidth"]), int(s["imHeight"]), float(s.get("frameRate", 25)))


def sportsmot_sequences(data_root: Path, split: str = "val", sport: str = "basketball") -> list[Sequence]:
    names = (data_root / "splits_txt" / f"{sport}.txt").read_text().split()
    out = []
    for n in names:
        p = data_root / split / n
        if p.exists():
            out.append(load_sequence(p))
    return out


def read_mot(path: Path) -> np.ndarray:
    """(N, >=6) rows: frame, id, x, y, w, h, [conf, ...]. Empty -> (0, 10)."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 10))
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    return a


def write_mot(path: Path, rows: list[tuple[int, int, float, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fr, tid, x1, y1, x2, y2, conf in rows:
            f.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{conf:.3f},-1,-1,-1\n")


def _by_frame(a: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    if len(a) == 0:
        return out
    for fr in np.unique(a[:, 0]).astype(int):
        out[fr] = a[a[:, 0] == fr]
    return out


def _xywh_to_xyxy(r: np.ndarray) -> np.ndarray:
    return np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], 1)


def hota(gt: np.ndarray, res: np.ndarray, alphas: np.ndarray | None = None) -> dict[str, float]:
    """HOTA, DetA, AssA (+ IDF1-style identity counts) following Luiten et al. 2021.

    gt/res rows: frame, id, x, y, w, h, ... ; gt rows with conf == 0 are ignored
    (MOTChallenge convention: column 7 flag). Frames are matched per alpha with
    the Hungarian algorithm on IoU, then association accuracy is computed from the
    global gt-id / pred-id co-occurrence counts of the matched pairs.
    """
    alphas = np.arange(0.05, 0.99, 0.05) if alphas is None else alphas
    if len(gt) and gt.shape[1] >= 7:
        gt = gt[gt[:, 6] != 0]
    G, R = _by_frame(gt), _by_frame(res)
    frames = sorted(set(G) | set(R))
    gt_ids = {int(i): k for k, i in enumerate(np.unique(gt[:, 1]))} if len(gt) else {}
    pr_ids = {int(i): k for k, i in enumerate(np.unique(res[:, 1]))} if len(res) else {}
    n_g, n_p = len(gt_ids), len(pr_ids)
    gt_count = np.zeros(n_g)
    pr_count = np.zeros(n_p)
    for fr in frames:
        for row in G.get(fr, []):
            gt_count[gt_ids[int(row[1])]] += 1
        for row in R.get(fr, []):
            pr_count[pr_ids[int(row[1])]] += 1
    total_gt, total_pr = gt_count.sum(), pr_count.sum()

    hota_a, deta_a, assa_a = [], [], []
    for a in alphas:
        tp = 0
        co = np.zeros((n_g, n_p))
        for fr in frames:
            g, r = G.get(fr), R.get(fr)
            if g is None or r is None or len(g) == 0 or len(r) == 0:
                continue
            iou = iou_matrix(_xywh_to_xyxy(g), _xywh_to_xyxy(r))
            rows, cols = linear_sum_assignment(-iou)
            for i, j in zip(rows, cols):
                if iou[i, j] >= a:
                    tp += 1
                    co[gt_ids[int(g[i, 1])], pr_ids[int(r[j, 1])]] += 1
        fn, fp = total_gt - tp, total_pr - tp
        deta = tp / max(tp + fn + fp, 1)
        if tp > 0:
            # association score per matched pair: TPA / (TPA + FNA + FPA)
            gi, pj = np.nonzero(co)
            tpa = co[gi, pj]
            fna = gt_count[gi] - tpa
            fpa = pr_count[pj] - tpa
            assa = float((tpa * (tpa / (tpa + fna + fpa))).sum() / tp)
        else:
            assa = 0.0
        hota_a.append(np.sqrt(deta * assa))
        deta_a.append(deta)
        assa_a.append(assa)
    return {"HOTA": float(np.mean(hota_a)), "DetA": float(np.mean(deta_a)), "AssA": float(np.mean(assa_a))}


def clear_mot(gt: np.ndarray, res: np.ndarray, iou_thresh: float = 0.5) -> dict[str, float]:
    """MOTA, IDF1 (via global id matching) and id switches at IoU 0.5."""
    if len(gt) and gt.shape[1] >= 7:
        gt = gt[gt[:, 6] != 0]
    G, R = _by_frame(gt), _by_frame(res)
    frames = sorted(set(G) | set(R))
    tp = fp = fn = idsw = 0
    last_match: dict[int, int] = {}
    gt_ids = {int(i): k for k, i in enumerate(np.unique(gt[:, 1]))} if len(gt) else {}
    pr_ids = {int(i): k for k, i in enumerate(np.unique(res[:, 1]))} if len(res) else {}
    co = np.zeros((len(gt_ids), len(pr_ids)))
    for fr in frames:
        g, r = G.get(fr), R.get(fr)
        ng, nr = 0 if g is None else len(g), 0 if r is None else len(r)
        if ng and nr:
            iou = iou_matrix(_xywh_to_xyxy(g), _xywh_to_xyxy(r))
            rows, cols = linear_sum_assignment(-iou)
            m = [(i, j) for i, j in zip(rows, cols) if iou[i, j] >= iou_thresh]
            tp += len(m)
            fp += nr - len(m)
            fn += ng - len(m)
            for i, j in m:
                gid, pid = int(g[i, 1]), int(r[j, 1])
                co[gt_ids[gid], pr_ids[pid]] += 1
                if gid in last_match and last_match[gid] != pid:
                    idsw += 1
                last_match[gid] = pid
        else:
            fp += nr
            fn += ng
    n_gt = sum(len(v) for v in G.values())
    mota = 1 - (fn + fp + idsw) / max(n_gt, 1)
    # IDF1: optimal one-to-one id matching maximising co-occurrence
    if co.size:
        rows, cols = linear_sum_assignment(-co)
        idtp = co[rows, cols].sum()
    else:
        idtp = 0.0
    n_pr = sum(len(v) for v in R.values())
    idf1 = 2 * idtp / max(n_gt + n_pr, 1)
    return {"MOTA": float(mota), "IDF1": float(idf1), "IDSW": int(idsw), "FP": int(fp), "FN": int(fn)}


def trackeval_hota(gt_file: Path, res_file: Path) -> dict[str, float] | None:
    """Reference HOTA via TrackEval, when installed. Returns None otherwise."""
    try:
        from trackeval.metrics import HOTA  # noqa: F401
    except Exception:
        return None
    # TrackEval needs its folder layout; we build the minimal in-memory data dict.
    from trackeval.metrics import HOTA as _H

    gt, res = read_mot(gt_file), read_mot(res_file)
    if len(gt) and gt.shape[1] >= 7:
        gt = gt[gt[:, 6] != 0]
    G, R = _by_frame(gt), _by_frame(res)
    n = int(max(list(G) + list(R) + [0]))
    gt_ids_map = {int(i): k for k, i in enumerate(np.unique(gt[:, 1]))}
    pr_ids_map = {int(i): k for k, i in enumerate(np.unique(res[:, 1]))}
    data = {
        "num_timesteps": n,
        "num_gt_ids": len(gt_ids_map),
        "num_tracker_ids": len(pr_ids_map),
        "num_gt_dets": int(len(gt)),
        "num_tracker_dets": int(len(res)),
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }
    for fr in range(1, n + 1):
        g, r = G.get(fr), R.get(fr)
        gi = np.array([gt_ids_map[int(x)] for x in (g[:, 1] if g is not None else [])], int)
        pi = np.array([pr_ids_map[int(x)] for x in (r[:, 1] if r is not None else [])], int)
        sim = iou_matrix(_xywh_to_xyxy(g), _xywh_to_xyxy(r)) if g is not None and r is not None else np.zeros((len(gi), len(pi)))
        data["gt_ids"].append(gi)
        data["tracker_ids"].append(pi)
        data["similarity_scores"].append(sim)
    out = _H().eval_sequence(data)
    return {"HOTA": float(np.mean(out["HOTA"])), "DetA": float(np.mean(out["DetA"])), "AssA": float(np.mean(out["AssA"]))}
