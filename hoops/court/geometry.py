"""Court model: line geometry in metres, named landmarks and drawable lines, per league.

Coordinate frame: origin at centre court, +x along the court length (towards the
"right" basket), +y across the width ("top" = far sideline in a broadcast). Units
are metres. Landmark ids and their ORDER are identical for every league, so a
keypoint model trained on one league's frames maps onto any court spec; only the
coordinates change. A landmark a league does not have (e.g. the three-point
"break" on the pre-2019 NCAA arc, which runs straight into the baseline) is NaN
and is skipped by the solver.

Specs (verified against rulebooks / multiple secondary sources, 2026-11-01):
  nba               94x50 ft, 3pt 23.75 ft (22 ft corners, 14.2 ft straight), 16 ft lane, 4 ft RA
  ncaa_men          94x50 ft, 3pt 22 ft 1.75 in (21 ft 7.875 in corners), 12 ft lane, 4 ft RA (2019-)
  ncaa_men_pre2019  94x50 ft, 3pt 20 ft 9 in uniform arc to the baseline, 12 ft lane
  ncaa_women        94x50 ft, 3pt 22 ft 1.75 in (2021-), 12 ft lane, 3 ft RA
  fiba              28x15 m, 3pt 6.75 m (0.9 m from sideline), 4.9 m lane, 1.25 m RA, FT 5.8 m
Use `set_court("ncaa_men")` to switch; `cur()` returns the active spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FT = 0.3048
_NAN = float("nan")


@dataclass(frozen=True)
class CourtSpec:
    name: str
    length: float
    width: float
    basket_from_baseline: float
    ft_line_from_baseline: float
    paint_half_width: float
    ft_circle_r: float
    centre_circle_r: float
    three_r: float
    three_corner_from_sideline: float | None  # None => arc runs to the baseline (no straight part)
    restricted_r: float
    hash_from_baseline: float | None  # 28 ft coaching-box / FIBA throw-in line; None if absent
    rim_floor_offset: float = 0.29  # floor point annotators mark under the rim, beyond basket centre

    # ---- derived ------------------------------------------------------
    @property
    def half_l(self) -> float:
        return self.length / 2

    @property
    def half_w(self) -> float:
        return self.width / 2

    @property
    def basket_x(self) -> float:
        return self.half_l - self.basket_from_baseline

    @property
    def ft_x(self) -> float:
        return self.half_l - self.ft_line_from_baseline

    @property
    def hash_x(self) -> float | None:
        return None if self.hash_from_baseline is None else self.half_l - self.hash_from_baseline

    @property
    def three_corner_y(self) -> float:
        """y of the straight corner segment, or of the arc/baseline meeting point."""
        if self.three_corner_from_sideline is not None:
            return self.half_w - self.three_corner_from_sideline
        return float(np.sqrt(self.three_r**2 - self.basket_from_baseline**2))

    @property
    def three_break_x(self) -> float | None:
        if self.three_corner_from_sideline is None:
            return None
        dx = float(np.sqrt(self.three_r**2 - self.three_corner_y**2))
        return self.basket_x - dx

    @property
    def rim_floor_x(self) -> float:
        return self.basket_x + self.rim_floor_offset


COURTS: dict[str, CourtSpec] = {
    "nba": CourtSpec("nba", 94 * FT, 50 * FT, 5.25 * FT, 19 * FT, 8 * FT, 6 * FT, 6 * FT, 23.75 * FT, 3 * FT, 4 * FT, 28 * FT),
    "ncaa_men": CourtSpec("ncaa_men", 94 * FT, 50 * FT, 5.25 * FT, 19 * FT, 6 * FT, 6 * FT, 6 * FT, (22 + 1.75 / 12) * FT, (25 - (21 + 7.875 / 12)) * FT, 4 * FT, 28 * FT),
    "ncaa_men_pre2019": CourtSpec("ncaa_men_pre2019", 94 * FT, 50 * FT, 5.25 * FT, 19 * FT, 6 * FT, 6 * FT, 6 * FT, (20 + 9 / 12) * FT, None, 4 * FT, 28 * FT),
    "ncaa_women": CourtSpec("ncaa_women", 94 * FT, 50 * FT, 5.25 * FT, 19 * FT, 6 * FT, 6 * FT, 6 * FT, (22 + 1.75 / 12) * FT, (25 - (21 + 7.875 / 12)) * FT, 3 * FT, 28 * FT),
    "fiba": CourtSpec("fiba", 28.0, 15.0, 1.575, 5.8, 2.45, 1.8, 1.8, 6.75, 0.9, 1.25, 8.325),
}


@dataclass(frozen=True)
class Landmark:
    id: str
    xy: tuple[float, float]
    group: str


def _sides(name: str, x: float | None, y: float, group: str) -> list[Landmark]:
    """L/R (and T/B when y != 0) mirrored copies; x None => landmark absent (NaN)."""
    out = []
    for sx, s in ((-1, "L"), (1, "R")):
        xv = _NAN if x is None else sx * x
        if abs(y) > 1e-9:
            for sy, t in ((1, "T"), (-1, "B")):
                out.append(Landmark(f"{s}{t}_{name}", (xv, sy * y), group))
        else:
            out.append(Landmark(f"{s}_{name}", (xv, y), group))
    return out


def build_landmarks(c: CourtSpec) -> list[Landmark]:
    """Same ids and order for every spec (order matters: it is the keypoint index space)."""
    L: list[Landmark] = []
    L += _sides("corner", c.half_l, c.half_w, "corner")
    L += _sides("paint_base", c.half_l, c.paint_half_width, "paint")
    L += _sides("paint_ft", c.ft_x, c.paint_half_width, "paint")
    L += _sides("ft_circle_apex", c.ft_x - c.ft_circle_r, 0.0, "ft")
    L += _sides("ft_mid", c.ft_x, 0.0, "ft")
    L += _sides("rim", c.rim_floor_x, 0.0, "rim")
    L += _sides("three_base", c.half_l, c.three_corner_y, "three")
    L += _sides("three_break", c.three_break_x, c.three_corner_y, "three")
    L += _sides("three_apex", c.basket_x - c.three_r, 0.0, "three")
    L += _sides("restricted_apex", c.basket_x - c.restricted_r, 0.0, "restricted")
    L += _sides("hash", c.hash_x, c.half_w, "hash")
    L.append(Landmark("T_half_side", (0.0, c.half_w), "centre"))
    L.append(Landmark("B_half_side", (0.0, -c.half_w), "centre"))
    L.append(Landmark("C_centre", (0.0, 0.0), "centre"))
    L.append(Landmark("CT_circle", (0.0, c.centre_circle_r), "centre"))
    L.append(Landmark("CB_circle", (0.0, -c.centre_circle_r), "centre"))
    L.append(Landmark("L_circle", (-c.centre_circle_r, 0.0), "centre"))
    L.append(Landmark("R_circle", (c.centre_circle_r, 0.0), "centre"))
    return L


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 40) -> np.ndarray:
    t = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)


def build_lines(c: CourtSpec) -> dict[str, np.ndarray]:
    """Named polylines (N, 2) in metres covering every painted line of the spec."""
    hl, hw = c.half_l, c.half_w
    lines: dict[str, np.ndarray] = {
        "sideline_T": np.array([[-hl, hw], [hl, hw]]),
        "sideline_B": np.array([[-hl, -hw], [hl, -hw]]),
        "baseline_L": np.array([[-hl, -hw], [-hl, hw]]),
        "baseline_R": np.array([[hl, -hw], [hl, hw]]),
        "half": np.array([[0.0, -hw], [0.0, hw]]),
        "centre_circle": _arc(0, 0, c.centre_circle_r, 0, 2 * np.pi, 80),
    }
    for sx, s in ((-1, "L"), (1, "R")):
        bx, fx, base, p = sx * c.basket_x, sx * c.ft_x, sx * hl, c.paint_half_width
        lines[f"paint_{s}"] = np.array([[base, -p], [fx, -p], [fx, p], [base, p]])
        lines[f"ft_circle_{s}"] = _arc(fx, 0, c.ft_circle_r, 0, 2 * np.pi, 80)
        y3 = c.three_corner_y
        if c.three_break_x is not None:
            bxk = sx * c.three_break_x
            lines[f"three_corner_{s}T"] = np.array([[base, y3], [bxk, y3]])
            lines[f"three_corner_{s}B"] = np.array([[base, -y3], [bxk, -y3]])
            dx = float(np.sqrt(c.three_r**2 - y3**2))
        else:
            dx = c.basket_from_baseline  # arc meets the baseline directly
        ang = float(np.arctan2(y3, -sx * dx))
        lines[f"three_arc_{s}"] = _arc(bx, 0, c.three_r, ang, 2 * np.pi - ang, 80) if sx == 1 else _arc(bx, 0, c.three_r, -ang, ang, 80)
        lines[f"restricted_{s}"] = _arc(bx, 0, c.restricted_r, np.pi / 2, 3 * np.pi / 2, 30) if sx == 1 else _arc(bx, 0, c.restricted_r, -np.pi / 2, np.pi / 2, 30)
        if c.hash_x is not None:
            hx = sx * c.hash_x
            lines[f"hash_{s}T"] = np.array([[hx, hw], [hx, hw - 0.9]])
            lines[f"hash_{s}B"] = np.array([[hx, -hw], [hx, -hw + 0.9]])
    return lines


# ---------------------------------------------------------------------------
# active court (module state). Arrays are updated IN PLACE so `from geometry import
# LANDMARK_XY` keeps working; scalars are read through cur() at call time.
# ---------------------------------------------------------------------------
_current: CourtSpec = COURTS["nba"]
LANDMARKS: list[Landmark] = build_landmarks(_current)
LANDMARK_IDS: list[str] = [lm.id for lm in LANDMARKS]
LANDMARK_XY: np.ndarray = np.array([lm.xy for lm in LANDMARKS], dtype=float)
_INDEX = {lm.id: i for i, lm in enumerate(LANDMARKS)}


def cur() -> CourtSpec:
    return _current


def set_court(name: str | CourtSpec) -> CourtSpec:
    """Switch the active court. Landmark ids/order are unchanged; coordinates update in place."""
    global _current, LANDMARKS
    spec = COURTS[name] if isinstance(name, str) else name
    _current = spec
    LANDMARKS = build_landmarks(spec)
    LANDMARK_XY[:] = np.array([lm.xy for lm in LANDMARKS], dtype=float)
    return spec


def landmark_index(id_: str) -> int:
    return _INDEX[id_]


def mirror_id(id_: str) -> str:
    """Landmark id after a horizontal (left<->right) flip of the court."""
    if id_.startswith("L"):
        return "R" + id_[1:]
    if id_.startswith("R"):
        return "L" + id_[1:]
    return id_


MIRROR_INDEX = np.array([_INDEX[mirror_id(i)] for i in LANDMARK_IDS])


def court_lines() -> dict[str, np.ndarray]:
    return build_lines(_current)


def inside_court(xy: np.ndarray, margin: float = 1.0) -> np.ndarray:
    xy = np.asarray(xy)
    return (np.abs(xy[..., 0]) <= _current.half_l + margin) & (np.abs(xy[..., 1]) <= _current.half_w + margin)


# NBA constants kept as module names for existing callers and tests (NBA values).
_nba = COURTS["nba"]
COURT_LENGTH, COURT_WIDTH = _nba.length, _nba.width
HALF_L, HALF_W = _nba.half_l, _nba.half_w
BASKET_FROM_BASELINE, FT_LINE_FROM_BASELINE = _nba.basket_from_baseline, _nba.ft_line_from_baseline
PAINT_HALF_WIDTH, FT_CIRCLE_R, CENTRE_CIRCLE_R = _nba.paint_half_width, _nba.ft_circle_r, _nba.centre_circle_r
THREE_R, THREE_CORNER_Y, RESTRICTED_R = _nba.three_r, _nba.three_corner_y, _nba.restricted_r
HASH_FROM_BASELINE, BASKET_X, FT_X, HASH_X = _nba.hash_from_baseline, _nba.basket_x, _nba.ft_x, _nba.hash_x
THREE_BREAK_X, RIM_FLOOR_X = _nba.three_break_x, _nba.rim_floor_x
