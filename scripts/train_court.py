"""Train the court landmark model (Ultralytics YOLO-pose, 22 Roboflow keypoints).

  uv run python scripts/train_court.py --epochs 80 --model yolo26n-pose.pt --imgsz 1280

Small dataset (56 train images): heavy augmentation, horizontal flips with the
left/right index swap, no mosaic (it breaks court geometry). Results land in
runs/court/<name>/weights/best.pt; use with
  hoops calibrate VIDEO --landmarks yolo:runs/court/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/court_yolo/court.yaml"))
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="court_pose")
    args = ap.parse_args()

    from ultralytics import YOLO

    from hoops.device import pick_device

    device = pick_device(args.device)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if device.startswith("cuda") else device,
        project=str(ROOT / "runs/court"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        mosaic=0.0,
        fliplr=0.5,
        degrees=3.0,
        translate=0.05,
        scale=0.3,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        patience=30,
        workers=4,
        plots=False,
        verbose=False,
    )
    best = ROOT / "runs/court" / args.name / "weights/best.pt"
    print("best weights:", best, best.exists())
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=device, plots=False, verbose=False)
    print({k: round(float(v), 4) for k, v in metrics.results_dict.items()})


if __name__ == "__main__":
    main()
