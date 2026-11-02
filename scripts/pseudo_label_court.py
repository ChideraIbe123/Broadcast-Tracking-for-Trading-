"""Bootstrap court-landmark labels for a new league from broadcast clips (self-training).

  uv run python scripts/pseudo_label_court.py --court ncaa_men_pre2019 --clips data/clips/v_00HRwkvvjtQ_c001.mp4 ... \
      --landmarks yolo:weights/court/court_bcd33_n960_best.pt@bcd33 --out data/court_yolo_ncaa_pseudo --holdout v_00HRwkvvjtQ_c011

For every frame the calibrator (landmarks -> RANSAC -> line-alignment refinement -> gate)
marks as trustworthy, we project the court model through the final homography and write
the in-frame landmarks as YOLO-pose labels in the 33-point BCD schema. Selection is strict:
line support of the *filtered* H above `--min-support`, and every physical check passed.
Labels inherit the calibration's error (~0.2-0.5 m, a few pixels) and ignore occlusion by
players; they are a bootstrap, not ground truth. One clip is held out for evaluation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hoops.court import geometry as g  # noqa: E402
from hoops.court.camera import CourtCalibrator  # noqa: E402
from hoops.court.homography import court_to_image  # noqa: E402
from hoops.court.keypoints import BCD33, BCD33_TO_HOOPS, make_landmark_source  # noqa: E402
from hoops.court.lines import edge_map, line_support  # noqa: E402
from hoops.ingest.cuts import detect_cuts, segments_from_cuts  # noqa: E402
from hoops.ingest.video import frames, probe  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--court", default="ncaa_men_pre2019")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--landmarks", default="yolo:weights/court/court_bcd33_n960_best.pt@bcd33")
    ap.add_argument("--out", default=str(ROOT / "data/court_yolo_ncaa_pseudo"))
    ap.add_argument("--holdout", default="", help="clip stem used for val instead of train")
    ap.add_argument("--min-support", type=float, default=0.05)
    ap.add_argument("--every", type=int, default=4, help="keep every Nth qualifying frame")
    ap.add_argument("--sheet", default=str(ROOT / "out/pseudo_label_sheet.jpg"))
    a = ap.parse_args()

    g.set_court(a.court)
    src = make_landmark_source(a.landmarks)
    lm_idx = np.array([g.landmark_index(BCD33_TO_HOOPS[n]) for n in BCD33])
    out = Path(a.out)
    for s in ("train", "val"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
    sheet = []
    totals = {"frames": 0, "kept": 0}
    for clip in a.clips:
        clip = Path(clip)
        split = "val" if clip.stem == a.holdout else "train"
        info = probe(clip)
        seg_of = np.zeros(info.n_frames + 1, int)
        for s in segments_from_cuts(detect_cuts(clip), info.n_frames):
            seg_of[s.start : s.end] = s.id
        cal = CourtCalibrator(info.shape)
        cur_seg, kept, n_q = -1, 0, 0
        for i, img in frames(clip):
            totals["frames"] += 1
            if seg_of[i] != cur_seg:
                cal.reset()
                cur_seg = seg_of[i]
            pts, idx, conf = src(i, img)
            res = cal.step(img, pts, idx, conf, None)
            if res.H is None or res.label == "rejected":
                continue
            s_f = line_support(res.H, edge_map(img))
            if s_f < a.min_support:
                continue
            n_q += 1
            if n_q % a.every:
                continue
            H, W = img.shape[:2]
            px = court_to_image(res.H, g.LANDMARK_XY[lm_idx])
            vis = np.isfinite(px).all(1) & (px[:, 0] > 2) & (px[:, 0] < W - 2) & (px[:, 1] > 2) & (px[:, 1] < H - 2)
            if vis.sum() < 5:
                continue
            us, vs = px[vis, 0], px[vis, 1]
            x1, y1, x2, y2 = max(us.min() - 10, 0), max(vs.min() - 10, 0), min(us.max() + 10, W), min(vs.max() + 10, H)
            parts = [0, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H]
            for k in range(len(BCD33)):
                parts += [px[k, 0] / W, px[k, 1] / H, 2] if vis[k] else [0, 0, 0]
            name = f"{clip.stem}_{i:06d}"
            cv2.imwrite(str(out / split / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / split / "labels" / f"{name}.txt").write_text(" ".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts))
            kept += 1
            totals["kept"] += 1
            if len(sheet) < 8 and kept % 10 == 1:
                vis_img = img.copy()
                for k in np.flatnonzero(vis):
                    cv2.circle(vis_img, (int(px[k, 0]), int(px[k, 1])), 5, (0, 0, 255), -1)
                cv2.putText(vis_img, f"{name} support {s_f:.3f} {res.label}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                sheet.append(cv2.resize(vis_img, (640, 360)))
        print(f"{clip.stem} ({split}): {kept} pseudo-labelled frames from {n_q} qualifying of {info.n_frames}", flush=True)
    if sheet:
        while len(sheet) % 2:
            sheet.append(np.zeros_like(sheet[0]))
        rows = [np.hstack(sheet[i : i + 2]) for i in range(0, len(sheet), 2)]
        Path(a.sheet).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    from hoops.court.keypoints import BCD33_FLIP_IDX

    (out / "court.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\nkpt_shape: [33, 3]\nflip_idx: {BCD33_FLIP_IDX}\nnames:\n  0: court\n"
    )
    print("total", totals, "->", out)


if __name__ == "__main__":
    main()
