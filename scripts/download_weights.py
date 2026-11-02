"""Download third-party weights into ./weights (idempotent).

  uv run python scripts/download_weights.py            # everything that runs on this machine
  uv run python scripts/download_weights.py --only wasb

Sources (verified 2026-10-28):
  rfdetr   : pulled automatically by the rfdetr package on first use (~/.roboflow/models)
  deep_eiou: YOLOX detector + OSNet sports ReID from the Deep-EIoU authors' Google Drive folder
             https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w
  wasb     : basketball ball-detection weights from nttcom/WASB-SBDT MODEL_ZOO (research-only licence)
  ultralytics ReID encoders download automatically (yolo26n-reid.onnx)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"

GDRIVE_FILES = {
    # Deep-EIoU folder contents (ids resolved at run time by listing the folder)
    "deep_eiou": {"folder": "https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w"},
    # WASB basketball weights (from MODEL_ZOO.md, column "basketball")
    "wasb": {
        "wasb_basketball.pth.tar": "https://drive.google.com/file/d/1nfECuSyJvPUmz3njZCdFERSQQbERt8FU/view?usp=drive_link",
        "tracknetv2_basketball.pth.tar": "https://drive.google.com/file/d/1n-R_T0QyENsArYV8Qn8TneEroQkmYbT6/view?usp=drive_link",
    },
}


def gdown_file(url: str, dest: Path) -> None:
    import gdown

    if dest.exists():
        print(f"  exists  {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetch   {dest.name}")
    gdown.download(url, str(dest), quiet=False, fuzzy=True)


def gdown_folder(url: str, dest: Path) -> None:
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"  exists  {dest}")
        return
    gdown.download_folder(url, output=str(dest), quiet=False, use_cookies=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of: rfdetr deep_eiou wasb ultralytics")
    args = ap.parse_args()
    want = set(args.only or ["rfdetr", "deep_eiou", "wasb", "ultralytics"])

    if "rfdetr" in want:
        print("rfdetr: warming the package cache (COCO Medium)")
        from rfdetr import RFDETRMedium

        RFDETRMedium(device="cpu")
    if "deep_eiou" in want:
        print("deep_eiou: YOLOX detector + OSNet sports ReID")
        gdown_folder(GDRIVE_FILES["deep_eiou"]["folder"], WEIGHTS / "deep_eiou")
    if "wasb" in want:
        print("wasb: basketball ball models")
        for name, url in GDRIVE_FILES["wasb"].items():
            gdown_file(url, WEIGHTS / "wasb" / name)
    if "ultralytics" in want:
        print("ultralytics: ReID encoder")
        from ultralytics.utils.downloads import attempt_download_asset

        (WEIGHTS / "ultralytics").mkdir(parents=True, exist_ok=True)
        attempt_download_asset("yolo26n-reid.onnx")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
