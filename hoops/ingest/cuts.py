"""Scene-cut detection for broadcast footage.

Broadcasts switch cameras constantly (replays, close-ups, crowd). A tracker that
does not know a cut happened will link a player in the old shot to a different
player in the new one, and the court homography from the old shot is meaningless
in the new one. We detect cuts with PySceneDetect's content detector and expose
them as segments; the tracker and calibration reset at every segment boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Segment:
    id: int
    start: int  # inclusive frame index
    end: int  # exclusive frame index

    def __contains__(self, frame: int) -> bool:
        return self.start <= frame < self.end


def detect_cuts(path: str | Path, threshold: float = 27.0, min_scene_len: int = 12) -> list[int]:
    """Return frame indices at which a new shot starts (excluding frame 0)."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(path), ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    cuts = [s[0].get_frames() for s in scenes[1:]]
    return sorted(set(int(c) for c in cuts))


def segments_from_cuts(cuts: list[int], n_frames: int) -> list[Segment]:
    bounds = [0] + [c for c in cuts if 0 < c < n_frames] + [n_frames]
    return [Segment(i, a, b) for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]


def segment_lookup(segments: list[Segment], n_frames: int) -> np.ndarray:
    """Array mapping frame index -> segment id."""
    out = np.zeros(n_frames, dtype=np.int32)
    for s in segments:
        out[s.start : s.end] = s.id
    return out
