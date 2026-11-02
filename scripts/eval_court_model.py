"""Evaluate a trained court landmark model through the full calibration path.

  uv run python scripts/eval_court_model.py runs/court/court_bcd33_n960/weights/best.pt \
      --schema bcd33 --data data/roboflow/bcd2_v21 --split test --unstretch 1280 720
  uv run python scripts/eval_court_model.py <weights> --schema bcd33 \
      --data data/roboflow/nba_court_keypoints --gt-schema roboflow22 --split test   # external test

For every labelled image: run the model, solve the homography (RANSAC + handedness
retry), apply the quality gate, then push the LABELLED landmarks through the recovered H
and measure their distance from the true court positions (metres). Reports median / p90
error, acceptance rate, and per-landmark pixel error. --gt-schema lets a model trained on
one keypoint schema be scored on a dataset labelled with another (via our landmark ids).
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
from hoops.court.gate import assess  # noqa: E402
from hoops.court.geometry import LANDMARK_IDS, LANDMARK_XY, landmark_index  # noqa: E402
from hoops.court.homography import HomographySolution, image_to_court, solve_homography, solve_homography_candidates  # noqa: E402
from hoops.court.keypoints import SCHEMAS, ModelLandmarks  # noqa: E402
from hoops.court.lines import select_by_line_support  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--schema", default="bcd33", help="keypoint order the model was trained with")
    ap.add_argument("--gt-schema", default=None, help="keypoint schema of the evaluation dataset (default: same)")
    ap.add_argument("--data", default=str(ROOT / "data/roboflow/bcd2_v21"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--unstretch", nargs=2, type=int, default=None, metavar=("W", "H"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--kind", default="yolo")
    ap.add_argument("--no-line-support", action="store_true", help="plain first RANSAC solution (ablation)")
    args = ap.parse_args()
    gt_schema = args.gt_schema or args.schema
    gt_names, gt_map, _ = SCHEMAS[gt_schema]

    src = Path(args.data) / ("valid" if args.split == "val" else args.split)
    ann = json.loads((src / "_annotations.coco.json").read_text())
    kp_cat = next(c for c in ann["categories"] if c.get("keypoints"))
    assert kp_cat["keypoints"] == gt_names, "dataset keypoint order != --gt-schema"
    by_img = {a["image_id"]: a["keypoints"] for a in ann["annotations"] if a["category_id"] == kp_cat["id"] and a.get("keypoints")}
    model = ModelLandmarks(args.weights, kind=args.kind, conf=args.conf, schema=args.schema)
    lm_of_gt = np.array([landmark_index(gt_map[n]) for n in gt_names])

    errs_m, labels, n_det = [], [], []
    px_err: dict[str, list[float]] = {n: [] for n in gt_names}
    for im in ann["images"]:
        if im["id"] not in by_img:
            continue
        img = cv2.imread(str(src / im["file_name"]))
        kps = np.array(by_img[im["id"]], float).reshape(-1, 3)
        if args.unstretch:
            W, H = args.unstretch
            kps[:, 0] *= W / im["width"]
            kps[:, 1] *= H / im["height"]
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        pts, idx, conf = model(0, img)
        n_det.append(len(idx))
        for k, (u, v, f) in enumerate(kps):
            if f > 0:
                hit = np.flatnonzero(idx == lm_of_gt[k])
                if len(hit):
                    px_err[gt_names[k]].append(float(np.hypot(*(pts[hit[0]] - [u, v]))))
        if args.no_line_support:
            sol = solve_homography(pts, idx, conf, img.shape[:2])
        else:
            cands = solve_homography_candidates(pts, idx, conf, img.shape[:2])
            chosen, _ = select_by_line_support(cands, img)
            sol = chosen if chosen is not None else HomographySolution(None, cands[0].inlier_mask, cands[0].residuals_px, cands[0].n_input, cands[0].spread, cands[0].reason)
        g = assess(sol, img.shape[:2])
        labels.append(g.label)
        if sol.ok and g.usable:
            lab = kps[:, 2] > 0
            rec = image_to_court(sol.H, kps[lab, :2])
            errs_m.append(np.median(np.linalg.norm(rec - LANDMARK_XY[lm_of_gt[lab]], axis=1)))
    errs_m = np.array(errs_m)
    n = len(labels)
    print(f"{args.split} ({Path(args.data).name}): {n} images, mean landmarks detected {np.mean(n_det):.1f}")
    print(f"gate: accepted {labels.count('accepted') / n:.0%}, uncertain {labels.count('uncertain') / n:.0%}, rejected {labels.count('rejected') / n:.0%}")
    if len(errs_m):
        print(f"court error on usable frames: median {np.median(errs_m):.2f} m, p90 {np.percentile(errs_m, 90):.2f} m, worst {errs_m.max():.2f} m, "
              f"frames < 1 m: {(errs_m < 1).mean():.0%} of usable")
    print("per-landmark pixel error (median, n):")
    for k, v in px_err.items():
        if v:
            print(f"  {k:20s} -> {LANDMARK_IDS[landmark_index(gt_map[k])]:16s} {np.median(v):6.1f} px  (n={len(v)})")


if __name__ == "__main__":
    main()
