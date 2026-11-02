"""Play-by-play loader -> per-event game state for win-probability training.

Source: shufinskiy/nba_data `nbastats_<season>.tar.xz` (stats.nba.com PBP v2,
seasons 1996-2025). One row per event:

    game_id, season, event_num, period, secs_left (whole game), home_margin,
    home_poss (1 home / 0 visitor / -1 unknown), home_win (label)

Conventions verified on the raw files: SCORE is "VISITOR - HOME", SCOREMARGIN is
home minus visitor ("TIE" = 0), PERSON1TYPE 4 = home player, 5 = visitor player.
Possession is reconstructed from event types (made shot / turnover flip it,
rebounds take it, jump balls go to PLAYER3), which is a decent approximation for a
game-level model.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "pbp" / "nba_data" / "datasets"
EXTRACTED = ROOT / "data" / "pbp" / "extracted"
CACHE = ROOT / "data" / "pbp" / "cache"

USECOLS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD", "PCTIMESTRING",
    "SCORE", "SCOREMARGIN", "PERSON1TYPE", "PLAYER1_TEAM_ID", "PERSON3TYPE", "PLAYER3_TEAM_ID",
]

# EVENTMSGTYPE codes
MADE, MISS, FT, REB, TOV, FOUL, VIOL, SUB, TIMEOUT, JUMP, EJECT, PSTART, PEND = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
LAST_FT_ACTIONS = {10, 12, 15, 16, 17, 19, 20, 22, 26, 29}  # "1 of 1", "2 of 2", "3 of 3", technical/flagrant last


def extract_season(season: int, kind: str = "nbastats") -> Path:
    csv = EXTRACTED / f"{kind}_{season}.csv"
    if csv.exists():
        return csv
    arc = RAW / f"{kind}_{season}.tar.xz"
    if not arc.exists():
        raise FileNotFoundError(f"{arc} (run scripts/download_datasets.py pbp)")
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    with tarfile.open(arc, "r:xz") as t:
        for m in t.getmembers():
            if m.name.endswith(".csv"):
                m.name = Path(m.name).name
                t.extract(m, EXTRACTED, filter="data")
    return csv


def _clock_to_secs(s: pd.Series) -> np.ndarray:
    mm = s.str.split(":", expand=True)
    return mm[0].astype(float).to_numpy() * 60 + mm[1].astype(float).to_numpy()


def _secs_left(period: np.ndarray, clock: np.ndarray) -> np.ndarray:
    reg = np.clip(4 - period, 0, None) * 720 + clock
    return np.where(period <= 4, reg, clock)  # in OT: seconds left in this OT period


def _possession(df: pd.DataFrame) -> np.ndarray:
    """home_poss after each event (1/0/-1), reconstructed per game."""
    out = np.full(len(df), -1, dtype=np.int8)
    for _, g in df.groupby("GAME_ID", sort=False):
        poss = -1
        for i, (etype, atype, p1, p3) in enumerate(zip(g.EVENTMSGTYPE, g.EVENTMSGACTIONTYPE, g.PERSON1TYPE, g.PERSON3TYPE)):
            team = 1 if p1 == 4 else 0 if p1 == 5 else -1  # home / visitor / unknown actor
            if etype in (MADE, TOV) and team >= 0:
                poss = 1 - team
            elif etype == MISS and team >= 0:
                poss = -1  # rebound decides
            elif etype == FT and team >= 0 and atype in LAST_FT_ACTIONS:
                poss = 1 - team  # assume made; a miss is followed by a rebound row that overrides
            elif etype == REB:
                if team >= 0:
                    poss = team
                elif p1 in (2, 3):  # team rebound rows: 2 = home team, 3 = visitor team
                    poss = 1 if p1 == 2 else 0
            elif etype == JUMP:
                poss = 1 if p3 == 4 else 0 if p3 == 5 else poss
            elif etype == PSTART:
                poss = -1
            out[g.index[i]] = poss
    return out


def load_season(season: int, regular_only: bool = True) -> pd.DataFrame:
    """Per-event game state for one season (cached as parquet)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"state_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    csv = extract_season(season)
    df = pd.read_csv(csv, usecols=USECOLS, dtype={"PCTIMESTRING": str, "SCORE": str, "SCOREMARGIN": str}, low_memory=False)
    if regular_only:
        df = df[df.GAME_ID.astype(str).str.startswith("2")]
    df = df.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"]).reset_index(drop=True)
    df["PCTIMESTRING"] = df.PCTIMESTRING.fillna("0:00")
    clock = _clock_to_secs(df.PCTIMESTRING)
    period = df.PERIOD.to_numpy()
    margin = df.SCOREMARGIN.replace("TIE", "0").astype(float)
    margin = margin.groupby(df.GAME_ID).ffill().fillna(0.0)
    df["home_margin"] = margin.to_numpy()
    df["secs_left"] = _secs_left(period, clock)
    df["home_poss"] = _possession(df)
    final = df.groupby("GAME_ID").home_margin.last()
    df["home_win"] = (df.GAME_ID.map(final) > 0).astype(np.int8)
    df = df[df.GAME_ID.map(final) != 0]  # drop games with broken scores
    out = pd.DataFrame(
        {
            "game_id": df.GAME_ID.astype(np.int64),
            "season": season,
            "event_num": df.EVENTNUM.astype(np.int32),
            "period": period.astype(np.int8)[df.index] if len(df) < len(period) else df.PERIOD.astype(np.int8),
            "secs_left": df.secs_left.astype(np.float32),
            "home_margin": df.home_margin.astype(np.float32),
            "home_poss": df.home_poss.astype(np.int8),
            "home_win": df.home_win,
        }
    ).reset_index(drop=True)
    out.to_parquet(cache, index=False)
    return out


def load_seasons(seasons: list[int]) -> pd.DataFrame:
    return pd.concat([load_season(s) for s in seasons], ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model features from the game state."""
    d = df.copy()
    t = d.secs_left.clip(lower=1.0)
    d["is_ot"] = (d.period > 4).astype(np.int8)
    d["margin_over_sqrt_t"] = d.home_margin / np.sqrt(t)
    d["margin_x_poss"] = d.home_margin * d.home_poss.replace(-1, 0)
    d["log_t"] = np.log(t)
    return d


FEATURES = ["secs_left", "home_margin", "home_poss", "period", "is_ot", "margin_over_sqrt_t", "margin_x_poss", "log_t"]
