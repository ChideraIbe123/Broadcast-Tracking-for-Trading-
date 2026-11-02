"""College (NCAA men's) schedules and scores from the sportsdataverse hoopR archive.

Files: data/college/mbb_schedule_<season>.parquet (season 2027 = 2026-27). Free, updated
daily upstream. We keep one row per game with ids, names, scores, neutral flag and date.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "college"
URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/schedules/parquet/mbb_schedule_{season}.parquet"
COLS = [
    "game_id", "season", "season_type", "game_date", "date", "neutral_site", "status_type_completed",
    "home_id", "home_location", "home_short_display_name", "home_display_name", "home_abbreviation", "home_score",
    "away_id", "away_location", "away_short_display_name", "away_display_name", "away_abbreviation", "away_score",
    "home_conference_id", "away_conference_id",
]


def fetch_season(season: int, refresh: bool = False) -> Path:
    DIR.mkdir(parents=True, exist_ok=True)
    f = DIR / f"mbb_schedule_{season}.parquet"
    if refresh or not f.exists():
        r = requests.get(URL.format(season=season), timeout=60)
        r.raise_for_status()
        f.write_bytes(r.content)
    return f


def load_games(seasons: list[int], refresh_current: bool = False, current_season: int | None = None) -> pd.DataFrame:
    """Games (completed and scheduled) for the given seasons, normalised."""
    frames = []
    for s in seasons:
        f = fetch_season(s, refresh=refresh_current and s == current_season)
        d = pd.read_parquet(f)
        d = d[[c for c in COLS if c in d.columns]].copy()
        frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    g["game_date"] = pd.to_datetime(g.game_date).dt.tz_localize(None)
    # tip-off time in UTC (ESPN 'date' field); falls back to noon UTC on the game date
    g["tip_utc"] = pd.to_datetime(g["date"], utc=True, errors="coerce") if "date" in g else pd.NaT
    g["tip_utc"] = g.tip_utc.fillna(pd.to_datetime(g.game_date).dt.tz_localize("UTC") + pd.Timedelta(hours=12))
    g["completed"] = g.status_type_completed.fillna(False).astype(bool)
    g["neutral"] = g.neutral_site.fillna(False).astype(bool)
    for c in ("home_id", "away_id"):
        g[c] = g[c].astype(str)
    g = g.dropna(subset=["home_id", "away_id"])
    g["margin"] = g.home_score - g.away_score
    g["total"] = g.home_score + g.away_score
    return g.sort_values("game_date").reset_index(drop=True)


def non_d1_teams(games: pd.DataFrame) -> set[str]:
    """Team ids that never carry a conference id: non-Division-I opponents (exhibition /
    buy games). They are ~30 points worse than an average D-I team and must not get the
    'average team' prior."""
    h = games[["home_id", "home_conference_id"]].rename(columns={"home_id": "id", "home_conference_id": "conf"})
    a = games[["away_id", "away_conference_id"]].rename(columns={"away_id": "id", "away_conference_id": "conf"})
    ids = pd.concat([h, a])
    flag = ids.groupby("id").conf.apply(lambda s: s.isna().all())
    return set(flag[flag].index.astype(str))


def team_table(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team id with the name variants used for matching Kalshi titles."""
    h = games[["home_id", "home_location", "home_short_display_name", "home_display_name", "home_abbreviation"]]
    a = games[["away_id", "away_location", "away_short_display_name", "away_display_name", "away_abbreviation"]]
    h.columns = a.columns = ["team_id", "location", "short", "display", "abbr"]
    t = pd.concat([h, a]).dropna(subset=["team_id"]).drop_duplicates("team_id")
    return t.reset_index(drop=True)
