"""Convert a Roboflow court keypoint export (COCO keypoints) to YOLO-pose format.

  uv run python scripts/convert_court_to_yolo.py --schema bcd33 --src data/roboflow/bcd2_v21 \
      --dst data/court_yolo_bcd33 --unstretch 1280 720
  uv run python scripts/convert_court_to_yolo.py --schema roboflow22   # the 80-image set

Roboflow exports of basketball-court-detection-2 are stretched to a square; --unstretch
resizes images back to the broadcast aspect ratio and rescales the keypoints, so the
model trains on the geometry it will see at inference. Writes dst/{train,val,test}/
{images,labels} and dst/court.yaml with kpt_shape and the left/right flip index.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hoops.court.keypoints import SCHEMAS  # noqa: E402

SPLITS = {"train": "train", "valid": "val", "test": "test"}


def convert(src: Path, dst: Path, schema: str, unstretch: tuple[int, int] | None) -> None:
    names, _, flip_idx = SCHEMAS[schema]
    for s_in, s_out in SPLITS.items():
        ann = json.loads((src / s_in / "_annotations.coco.json").read_text())
        kp_cat = next(c for c in ann["categories"] if c.get("keypoints"))
        assert kp_cat["keypoints"] == names, f"keypoint order differs from schema {schema}"
        (dst / s_out / "images").mkdir(parents=True, exist_ok=True)
        (dst / s_out / "labels").mkdir(parents=True, exist_ok=True)
        by_img: dict[int, list] = {}
        for a in ann["annotations"]:
            if a["category_id"] == kp_cat["id"] and a.get("keypoints"):
                by_img.setdefault(a["image_id"], []).append(a)
        n = 0
        for im in ann["images"]:
            W, H = im["width"], im["height"]
            src_img = src / s_in / im["file_name"]
            dst_img = dst / s_out / "images" / im["file_name"]
            if unstretch:
                img = cv2.imread(str(src_img))
                cv2.imwrite(str(dst_img), cv2.resize(img, unstretch, interpolation=cv2.INTER_AREA if unstretch[0] < W else cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                shutil.copy(src_img, dst_img)
            lines = []
            for a in by_img.get(im["id"], []):
                kps = a["keypoints"]
                vis = [(kps[3 * i] / W, kps[3 * i + 1] / H, int(kps[3 * i + 2])) for i in range(len(kps) // 3)]  # normalised
                labelled = [(u, v) for u, v, f in vis if f > 0]
                if len(labelled) < 4:
                    continue
                us, vs = zip(*labelled)
                pad = 0.01
                x1, y1, x2, y2 = max(min(us) - pad, 0), max(min(vs) - pad, 0), min(max(us) + pad, 1), min(max(vs) + pad, 1)
                parts = [0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                for u, v, f in vis:
                    parts += [min(max(u, 0), 1), min(max(v, 0), 1), 2 if f == 2 else 1] if f > 0 else [0, 0, 0]
                lines.append(" ".join(f"{p:.6f}" if isinstance(p, float) else str(p) for p in parts))
                n += 1
            (dst / s_out / "labels" / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        print(f"{s_out}: {len(ann['images'])} images, {n} court instances")
    yaml = f"""path: {dst.resolve()}
train: train/images
val: val/images
test: test/images
kpt_shape: [{len(names)}, 3]
flip_idx: {flip_idx}
names:
  0: court
"""
    (dst / "court.yaml").write_text(yaml)
    print("wrote", dst / "court.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="bcd33", choices=list(SCHEMAS))
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--unstretch", nargs=2, type=int, default=None, metavar=("W", "H"))
    a = ap.parse_args()
    defaults = {
        "bcd33": (ROOT / "data/roboflow/bcd2_v21", ROOT / "data/court_yolo_bcd33"),
        "roboflow22": (ROOT / "data/roboflow/nba_court_keypoints", ROOT / "data/court_yolo"),
    }
    src = Path(a.src) if a.src else defaults[a.schema][0]
    dst = Path(a.dst) if a.dst else defaults[a.schema][1]
    convert(src, dst, a.schema, tuple(a.unstretch) if a.unstretch else None)
