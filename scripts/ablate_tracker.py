"""Ablate HoopsTracker settings on cached SportsMOT detections.

  uv run python scripts/ablate_tracker.py --max-seqs 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hoops.eval.bench import cached_detections, run_tracker_on_sequence
from hoops.eval.mot import clear_mot, hota, read_mot, sportsmot_sequences, write_mot
from hoops.registry import make_detector
from hoops.track.ours import HoopsTracker, HoopsTrackerConfig

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "cv default": dict(),
    "cv gate_xy x1": dict(use_gate=True),
    "cv gate_xy x3": dict(use_gate=True, gate_multiplier=3.0),
    "cv gate_xy x10": dict(use_gate=True, gate_multiplier=10.0),
    "cv no_cmc": dict(cmc=None),
    "cv eiou1.2": dict(iou_expand=1.2),
    "cv thr0.4/0.5": dict(high_thresh=0.4, new_track_thresh=0.5),
    "cv match0.9": dict(match_thresh=0.9),
    "imm": dict(motion="imm"),
    "ca": dict(motion="kalman_ca"),
    "particle": dict(motion="particle", motion_kw=dict(sigma_meas=6.0, sigma_acc=1.5, sigma_init_pos=4.0, sigma_init_vel=4.0, n_particles=300)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/sportsmot"))
    ap.add_argument("--max-seqs", type=int, default=2)
    ap.add_argument("--detector", default="rfdetr:medium")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    seqs = sportsmot_sequences(Path(args.data), "val")[: args.max_seqs]
    cache = ROOT / "out/bench/detcache"
    det = None
    dets = {}
    for s in seqs:
        f = cache / f"{args.detector.replace(':', '_')}__{s.name}.npz"
        if not f.exists() and det is None:
            det = make_detector(args.detector)
        dets[s.name] = cached_detections(s, args.detector, det, cache)
    print(f"{'variant':26s} {'HOTA':>6s} {'DetA':>6s} {'AssA':>6s} {'IDF1':>6s} {'MOTA':>6s} {'IDSW':>5s}")
    for name, kw in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        tr = HoopsTracker(HoopsTrackerConfig(**kw))
        ms = []
        for s in seqs:
            rows, _ = run_tracker_on_sequence(s, dets[s.name], tr, needs_frames=tr.cfg.cmc is not None)
            rf = ROOT / "out/ablate" / name.replace(" ", "_") / f"{s.name}.txt"
            write_mot(rf, rows)
            gt, res = read_mot(s.gt_file), read_mot(rf)
            ms.append({**hota(gt, res), **clear_mot(gt, res)})
        agg = {k: np.mean([m[k] for m in ms]) for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")}
        idsw = sum(m["IDSW"] for m in ms)
        print(f"{name:26s} {agg['HOTA']:6.3f} {agg['DetA']:6.3f} {agg['AssA']:6.3f} {agg['IDF1']:6.3f} {agg['MOTA']:6.3f} {idsw:5d}", flush=True)


if __name__ == "__main__":
    main()
