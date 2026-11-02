"""Fetch the public datasets into ./data (idempotent). Nothing here is committed.

  uv run python scripts/download_datasets.py sportvu     # 2015-16 NBA tracking mirror (~3.7 GB, git clone)
  uv run python scripts/download_datasets.py pbp         # play-by-play 1996-2025 (shufinskiy/nba_data)
  uv run python scripts/download_datasets.py roboflow    # needs ROBOFLOW_API_KEY
  uv run python scripts/download_datasets.py sportsmot   # prints the manual steps (CodaLab registration)

SportsMOT (ICCV 2023) requires registering for the CodaLab competition; the data
may not be redistributed. After downloading, place the extracted folder at
data/sportsmot/ so that data/sportsmot/splits_txt/basketball.txt exists.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ROBOFLOW_DATASETS = {
    # workspace, project, version -> local folder
    "basketball_detection": ("roboflow-jvuqo", "basketball-player-detection-3-ycjdo", None),
    # 80-image, 22-keypoint set (CC BY 4.0): kept as an external test set for calibration
    "nba_court_keypoints": ("sportanalytics-sykyt", "nba-court-keypoint-detection-u9n5v-gwoaa", None),
    # 850-frame, 33-keypoint set (CC BY 4.0), v21 = 960x960 stretch, no augmentation; the
    # converter un-stretches it to 1280x720. Main court-landmark training set.
    "bcd2_v21": ("roboflow-jvuqo", "basketball-court-detection-2", 21),
}


def sportvu() -> None:
    dest = DATA / "sportvu"
    if dest.exists():
        print("exists", dest)
        return
    subprocess.check_call(
        ["git", "clone", "--depth", "1", "https://github.com/linouk23/NBA-Player-Movements", str(dest)]
    )


def pbp() -> None:
    dest = DATA / "pbp" / "nba_data"
    if dest.exists():
        print("exists", dest)
        return
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/shufinskiy/nba_data", str(dest)])
    print("See", dest / "README.md", "for the per-season archive links; hoops.wp.data_pbp loads them.")


def roboflow() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("set ROBOFLOW_API_KEY (free account) to download Roboflow Universe datasets")
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    for name, (ws, proj, ver) in ROBOFLOW_DATASETS.items():
        dest = DATA / "roboflow" / name
        if dest.exists():
            print("exists", dest)
            continue
        p = rf.workspace(ws).project(proj)
        v = p.version(ver) if ver else p.versions()[0]
        v.download("coco", location=str(dest))


def sportsmot() -> None:
    print(__doc__)
    print("1. Register: https://codalab.lisn.upsaclay.fr/competitions/12424 (SportsMOT)")
    print("2. Download train/val/test + splits_txt from Participate > Get Data")
    print("3. Extract to", DATA / "sportsmot")


if __name__ == "__main__":
    which = sys.argv[1:] or ["sportsmot"]
    for w in which:
        {"sportvu": sportvu, "pbp": pbp, "roboflow": roboflow, "sportsmot": sportsmot}[w]()
