"""Fine-tune RF-DETR on the Roboflow basketball dataset (10 classes, COCO/Roboflow layout).

  uv run python scripts/train_detector.py --size medium --epochs 30          # A100
  uv run python scripts/train_detector.py --size small --epochs 2 --smoke    # quick check on the Mac

Output: runs/detector/<size>/checkpoint_best_total.pth (+ EMA variants). Use with
  hoops track VIDEO --detector rfdetr:medium --weights runs/detector/medium/checkpoint_best_total.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/roboflow/basketball_detection"))
    ap.add_argument("--size", default="medium", choices=SIZES)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=None, help="e.g. 704 for more small-object (ball) recall")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    ap.add_argument("--resume", default=None, help="Lightning .ckpt to resume from (e.g. runs/detector/medium/last.ckpt)")
    ap.add_argument("--name", default=None, help="output folder under runs/detector (default: the size)")
    args = ap.parse_args()

    import rfdetr

    from hoops.device import pick_device

    device = pick_device(args.device)
    model = getattr(rfdetr, SIZES[args.size])(device=device)
    out = ROOT / "runs/detector" / (args.name or args.size)
    kw = dict(
        dataset_dir=args.data,
        dataset_file="roboflow",
        output_dir=str(out),
        epochs=1 if args.smoke else args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        device=device,
        tensorboard=False,
        early_stopping=not args.smoke,
        early_stopping_patience=8,
        num_workers=2,
        checkpoint_interval=5,
    )
    if args.resolution:
        kw["resolution"] = args.resolution
    if args.resume:
        kw["resume"] = args.resume
    if device == "mps":
        kw["amp_dtype"] = None  # no mixed precision on Metal
    print("training", args.size, "on", device, "->", out)
    model.train(**kw)
    print("checkpoints:", sorted(p.name for p in out.glob("*.pth")))


if __name__ == "__main__":
    main()
