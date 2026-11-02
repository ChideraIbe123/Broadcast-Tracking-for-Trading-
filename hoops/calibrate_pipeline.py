"""Video + landmark source (+ tracks) -> per-frame homography, gate label, court coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hoops.court.camera import CourtCalibrator
from hoops.court.gate import foot_points_to_court
from hoops.court.keypoints import LandmarkSource
from hoops.ingest.cuts import detect_cuts, segments_from_cuts
from hoops.ingest.video import VideoWriter, frames, probe
from hoops.viz.minimap import Minimap
from hoops.viz.overlay import draw_court, draw_status


@dataclass
class CalibRun:
    calib_npz: Path
    court_parquet: Path | None
    overlay_mp4: Path | None
    n_frames: int
    accepted: int
    uncertain: int
    rejected: int


def calibrate_video(
    video: str | Path,
    source: LandmarkSource,
    out_dir: str | Path,
    tracks_parquet: str | Path | None = None,
    overlay: bool = True,
    detect_scene_cuts: bool = True,
    start: int = 0,
    end: int | None = None,
    progress: bool = True,
) -> CalibRun:
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    end = info.n_frames if end is None else min(end, info.n_frames)
    cuts = detect_cuts(video) if detect_scene_cuts else []
    seg_of = np.zeros(info.n_frames + 1, int)
    for s in segments_from_cuts(cuts, info.n_frames):
        seg_of[s.start : s.end] = s.id

    tracks = pd.read_parquet(tracks_parquet) if tracks_parquet else None
    by_frame = {f: g for f, g in tracks.groupby("frame")} if tracks is not None else {}

    cal = CourtCalibrator(info.shape)
    mini = Minimap()
    writer = VideoWriter(out_dir / f"{video.stem}_calib.mp4", info.fps, info.shape) if overlay else None
    Hs, labels, sources, frame_ids = [], [], [], []
    court_rows = []
    counts = {"accepted": 0, "uncertain": 0, "rejected": 0}
    cur_seg = -1
    it = frames(video, start, end)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, total=end - start, desc="calibrate", unit="f")
    for idx, img in it:
        seg = int(seg_of[idx])
        if seg != cur_seg:
            cal.reset()
            cur_seg = seg
        pts, lid, conf = source(idx, img)
        g = by_frame.get(idx)
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy() if g is not None else None
        out = cal.step(img, pts, lid, conf, boxes)
        counts[out.label] += 1
        Hs.append(out.H if out.H is not None else np.full((3, 3), np.nan))
        labels.append(out.label)
        sources.append(out.source)
        frame_ids.append(idx)
        if g is not None and out.H is not None:
            xy = foot_points_to_court(out.H, boxes)
            for (_, r), (x, y) in zip(g.iterrows(), xy):
                court_rows.append({"frame": idx, "segment": seg, "track_id": int(r.track_id), "x_m": float(x), "y_m": float(y), "gate": out.label})
        if writer is not None:
            if out.H is not None:
                draw_court(img, out.H, color=(0, 255, 255) if out.label == "accepted" else (0, 165, 255))
                if g is not None:
                    m = mini.render(foot_points_to_court(out.H, boxes), g.track_id.tolist())
                    mini.attach(img, m)
            draw_status(img, f"f{idx} seg{seg} {out.label} ({out.source}) {' | '.join(out.reasons[:2])}", out.label == "accepted")
            writer.write(img)
    if writer is not None:
        writer.close()
    npz = out_dir / f"{video.stem}_calib.npz"
    np.savez(npz, frame=np.array(frame_ids), H=np.stack(Hs), label=np.array(labels), source=np.array(sources))
    pq = None
    if court_rows:
        pq = out_dir / f"{video.stem}_court.parquet"
        pd.DataFrame(court_rows).to_parquet(pq, index=False)
    return CalibRun(npz, pq, None if writer is None else writer.path, len(frame_ids), counts["accepted"], counts["uncertain"], counts["rejected"])
