"""End-to-end: video -> cuts -> detections -> tracks (-> court coords) -> Parquet + overlay."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hoops.detect.base import Detections, Detector
from hoops.ingest.cuts import Segment, detect_cuts, segments_from_cuts
from hoops.ingest.video import VideoWriter, frames, probe
from hoops.track.base import Tracker, tracks_to_rows
from hoops.viz.overlay import draw_status, draw_tracks


@dataclass
class TrackRun:
    video: Path
    out_dir: Path
    segments: list[Segment]
    tracks_parquet: Path
    overlay_mp4: Path | None
    fps_processed: float
    n_frames: int
    timings: dict[str, float] = field(default_factory=dict)


def track_video(
    video: str | Path,
    detector: Detector,
    tracker: Tracker,
    out_dir: str | Path,
    detect_scene_cuts: bool = True,
    overlay: bool = True,
    start: int = 0,
    end: int | None = None,
    stride: int = 1,
    progress: bool = True,
) -> TrackRun:
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    end = info.n_frames if end is None else min(end, info.n_frames)

    t0 = time.time()
    cuts = detect_cuts(video) if detect_scene_cuts else []
    segments = segments_from_cuts(cuts, info.n_frames)
    t_cuts = time.time() - t0
    seg_of = np.zeros(info.n_frames + 1, int)
    for s in segments:
        seg_of[s.start : s.end] = s.id

    rows: list[dict] = []
    writer = VideoWriter(out_dir / f"{video.stem}_tracks.mp4", info.fps / stride, info.shape) if overlay else None
    det_time = trk_time = 0.0
    n = 0
    cur_seg = -1
    it = frames(video, start, end, stride)
    if progress:
        from tqdm import tqdm

        it = tqdm(it, total=(end - start) // stride, desc="track", unit="f")
    for idx, img in it:
        seg = int(seg_of[idx])
        if seg != cur_seg:
            tracker.reset()
            cur_seg = seg
        t = time.time()
        dets: Detections = detector(img)
        det_time += time.time() - t
        t = time.time()
        tracks = tracker.update(dets, img)
        trk_time += time.time() - t
        rows += tracks_to_rows(idx, seg, tracks)
        if writer is not None:
            draw_tracks(img, tracks)
            draw_status(img, f"f{idx} seg{seg} {len(tracks)} tracks | {getattr(tracker, 'name', '')}")
            writer.write(img)
        n += 1
    if writer is not None:
        writer.close()
    total = time.time() - t0

    df = pd.DataFrame(rows)
    pq = out_dir / f"{video.stem}_tracks.parquet"
    df.to_parquet(pq, index=False)
    return TrackRun(
        video,
        out_dir,
        segments,
        pq,
        None if writer is None else writer.path,
        n / max(total, 1e-9),
        n,
        {"cuts_s": t_cuts, "detect_s": det_time, "track_s": trk_time, "total_s": total},
    )
