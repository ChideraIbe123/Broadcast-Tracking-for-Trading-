"""Pick diverse frames from clips for manual court-landmark labelling and pre-fill them.

  uv run python scripts/select_label_frames.py --clips data/clips/v_00HRwkvvjtQ_c0*.mp4 --n 200 --court ncaa_men_pre2019

Writes out/label_frames/<name>.jpg plus prelabels.json:
  {"<name>": {"court": "...", "points": {"<landmark id>": [u, v], ...}}}
Pre-filled points come from the current landmark model (so a labeller corrects rather
than places) and are only a starting guess. Frames are sampled evenly across each clip
and across camera segments, skipping near-duplicates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hoops.court import geometry as g  # noqa: E402
from hoops.court.keypoints import make_landmark_source  # noqa: E402
from hoops.ingest.video import frames, probe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--court", default="ncaa_men_pre2019")
    ap.add_argument("--landmarks", default="yolo:weights/court/court_bcd33_n960_best.pt@bcd33")
    ap.add_argument("--out", default=str(ROOT / "out/label_frames"))
    a = ap.parse_args()
    g.set_court(a.court)
    src = make_landmark_source(a.landmarks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(probe(c).n_frames for c in a.clips)
    pre: dict[str, dict] = {}
    for clip in a.clips:
        clip = Path(clip)
        info = probe(clip)
        k = max(int(round(a.n * info.n_frames / total)), 1)
        picks = set(np.linspace(0, info.n_frames - 1, k).astype(int).tolist())
        prev_small = None
        for i, img in frames(clip):
            if i not in picks:
                continue
            small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (32, 18)).astype(float)
            if prev_small is not None and np.corrcoef(small.ravel(), prev_small.ravel())[0, 1] > 0.995:
                continue  # near-duplicate of the previous pick
            prev_small = small
            name = f"{clip.stem}_{i:06d}"
            cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            pts, idx, conf = src(i, img)
            pre[name] = {"court": a.court, "size": [img.shape[1], img.shape[0]], "points": {g.LANDMARK_IDS[int(l)]: [round(float(p[0]), 1), round(float(p[1]), 1)] for p, l in zip(pts, idx)}}
    (out / "prelabels.json").write_text(json.dumps(pre, indent=1))
    print(f"{len(pre)} frames -> {out}  (prelabels.json has model guesses for {sum(len(v['points']) for v in pre.values())} points)")


if __name__ == "__main__":
    main()
