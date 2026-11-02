"""Benchmark harness: every (detector, tracker, motion model) on SportsMOT basketball.

Detections are computed once per (detector, sequence) and cached as .npz so all
trackers see identical inputs; the comparison then isolates the tracker.
Results go to out/bench/<split>/<tracker>/<seq>.txt (MOT format) plus a JSON and
a Markdown table (BENCHMARKS.md is updated by `scripts/update_benchmarks.py`).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from hoops.detect.base import Detections
from hoops.eval.mot import Sequence, clear_mot, hota, read_mot, sportsmot_sequences, trackeval_hota, write_mot
from hoops.registry import make_detector, make_embedder, make_tracker


def cached_detections(seq: Sequence, detector_name: str, detector, cache_dir: Path) -> list[Detections]:
    f = cache_dir / f"{detector_name.replace(':', '_')}__{seq.name}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        names = z["names"].item()
        out = []
        for i in range(seq.n_frames):
            m = z["frame"] == i + 1
            out.append(Detections(z["xyxy"][m], z["conf"][m], z["cls"][m], names))
        return out
    xyxy, conf, cls, frame = [], [], [], []
    names = {}
    for i in range(1, seq.n_frames + 1):
        img = cv2.imread(str(seq.frame_path(i)))
        d = detector(img)
        names = d.names
        xyxy.append(d.xyxy)
        conf.append(d.conf)
        cls.append(d.cls)
        frame.append(np.full(len(d), i))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        f,
        xyxy=np.concatenate(xyxy) if xyxy else np.zeros((0, 4), np.float32),
        conf=np.concatenate(conf) if conf else np.zeros(0, np.float32),
        cls=np.concatenate(cls) if cls else np.zeros(0, int),
        frame=np.concatenate(frame) if frame else np.zeros(0, int),
        names=np.array(names, dtype=object),
    )
    return cached_detections(seq, detector_name, detector, cache_dir)


def run_tracker_on_sequence(seq: Sequence, dets: list[Detections], tracker, needs_frames: bool) -> tuple[list, float]:
    tracker.reset()
    rows = []
    t0 = time.time()
    for i, d in enumerate(dets, start=1):
        img = cv2.imread(str(seq.frame_path(i))) if needs_frames else None
        # SportsMOT ground truth is players only
        d = d.by_name("player") if "player" in d.names.values() else d
        for t in tracker.update(d, img):
            rows.append((i, t.id, *t.xyxy.tolist(), t.conf))
    return rows, len(dets) / max(time.time() - t0, 1e-9)


def run_bench(
    data_root: Path,
    split: str,
    trackers: list[str],
    detector_name: str,
    reid: str,
    conf: float,
    device: str | None,
    max_seqs: int | None,
    out_dir: Path,
    weights: Path | None = None,
) -> list[dict]:
    seqs = sportsmot_sequences(Path(data_root), split)
    if not seqs:
        raise SystemExit(f"no basketball sequences under {data_root}/{split}; see scripts/download_datasets.py sportsmot")
    if max_seqs:
        seqs = seqs[:max_seqs]
    detector = make_detector(detector_name, device=device, conf=conf, weights=weights)
    cache_key = detector_name + (f"+{Path(weights).parent.name}_{Path(weights).stem}" if weights else "")
    cache = Path(out_dir) / "detcache"
    all_dets = {s.name: cached_detections(s, cache_key, detector, cache) for s in seqs}

    results = []
    for tname in trackers:
        embedder = make_embedder(reid, device=device)
        tracker = make_tracker(tname, embedder=embedder, device=device)
        needs_frames = embedder is not None or tname.split(":")[0] in ("trackers", "boxmot", "ultralytics") or "hoops" in tname
        per_seq, fps = [], []
        for s in seqs:
            rows, f = run_tracker_on_sequence(s, all_dets[s.name], tracker, needs_frames)
            res_file = Path(out_dir) / split / tname.replace(":", "_") / f"{s.name}.txt"
            write_mot(res_file, rows)
            gt, res = read_mot(s.gt_file), read_mot(res_file)
            m = {**hota(gt, res), **clear_mot(gt, res)}
            ref = trackeval_hota(s.gt_file, res_file)
            if ref:
                m["HOTA_trackeval"] = ref["HOTA"]
            per_seq.append(m)
            fps.append(f)
        agg = {k: float(np.mean([m[k] for m in per_seq])) for k in per_seq[0] if k != "IDSW"}
        agg["IDSW"] = int(sum(m["IDSW"] for m in per_seq))
        agg["fps"] = float(np.mean(fps))
        agg["tracker"] = tname
        agg["n_seqs"] = len(seqs)
        results.append(agg)
        print(f"{tname:24s} HOTA {agg['HOTA']:.3f} IDF1 {agg['IDF1']:.3f} MOTA {agg['MOTA']:.3f} IDSW {agg['IDSW']:4d} fps {agg['fps']:.1f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"results_{split}.json").write_text(
        json.dumps({"detector": detector_name, "weights": str(weights) if weights else None, "reid": reid, "results": results}, indent=2)
    )
    return results
