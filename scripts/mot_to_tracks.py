"""Convert a MOT-format result file (from `hoops bench`) to the tracks parquet that
`hoops calibrate --tracks` expects, so benchmark output can feed the calibrator.

  uv run python scripts/mot_to_tracks.py out/bench/val/hoops_kalman_cv/v_00HRwkvvjtQ_c001.txt out/clips/v_00HRwkvvjtQ_c001_tracks.parquet
MOT frames are 1-based; video frames are 0-based.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
a = np.loadtxt(src, delimiter=",", ndmin=2)
df = pd.DataFrame(
    {
        "frame": a[:, 0].astype(int) - 1,
        "segment": 0,
        "track_id": a[:, 1].astype(int),
        "cls": 0,
        "x1": a[:, 2],
        "y1": a[:, 3],
        "x2": a[:, 2] + a[:, 4],
        "y2": a[:, 3] + a[:, 5],
        "conf": a[:, 6] if a.shape[1] > 6 else 1.0,
        "state": "tracked",
    }
)
dst.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(dst, index=False)
print(f"{len(df)} rows, {df.track_id.nunique()} ids, frames {df.frame.min()}-{df.frame.max()} -> {dst}")
