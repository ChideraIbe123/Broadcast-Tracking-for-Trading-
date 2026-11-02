"""Match Kalshi NCAA market titles to schedule games and score them with the rating model.

Kalshi: event ticker KXNCAAMBGAME-26NOV05TOWSUTAH has the date and two team codes; the
two markets of the event carry the team names in yes_sub_title ("Utah", "Towson").
Spread markets: "Texas wins by over 17.5 points" (floor_strike = 17.5); totals:
"Full Game: Over 167.5 points scored". Spread/total events share the date+codes with
the game event, which is how we recover both team names for them.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import pandas as pd

from .ratings import RatingModel

ALIASES = {  # Kalshi name -> ESPN location/short name when they differ
    "UConn": "UConn", "Ole Miss": "Ole Miss", "Miami (FL)": "Miami", "Miami (OH)": "Miami (OH)",
    "St. John's": "St. John's", "LSU": "LSU", "USC": "USC", "UCLA": "UCLA", "SMU": "SMU", "TCU": "TCU", "BYU": "BYU",
    "UNLV": "UNLV", "VCU": "VCU", "UCF": "UCF", "UAB": "UAB", "UTEP": "UTEP", "UTSA": "UTSA", "NC State": "NC State",
    "Saint Mary's": "Saint Mary's", "Pitt": "Pittsburgh", "Penn": "Pennsylvania", "UMass": "UMass",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("state", "st").replace("saint", "st"))


class TeamMatcher:
    def __init__(self, teams: pd.DataFrame):
        self.teams = teams
        self.by_norm: dict[str, str] = {}
        for r in teams.itertuples():
            for name in (r.location, r.short, r.display):
                if isinstance(name, str):
                    self.by_norm.setdefault(_norm(name), r.team_id)
        self.keys = list(self.by_norm)

    def match(self, kalshi_name: str) -> str | None:
        name = ALIASES.get(kalshi_name, kalshi_name)
        k = _norm(name)
        if k in self.by_norm:
            return self.by_norm[k]
        close = difflib.get_close_matches(k, self.keys, n=1, cutoff=0.9)
        return self.by_norm[close[0]] if close else None


@dataclass
class ScoredMarket:
    ticker: str
    kind: str  # game | spread | total
    title: str
    game_date: str
    home: str
    away: str
    neutral: bool
    yes_prob: float  # model probability the YES side settles
    yes_bid: float
    yes_ask: float
    edge_yes: float  # yes_prob - yes_ask
    edge_no: float  # (1 - yes_prob) - (1 - yes_bid)
    matched: bool
    note: str = ""


def _event_parts(event_ticker: str) -> tuple[str, str] | None:
    m = re.match(r"^KX\w+?-(\d{2}[A-Z]{3}\d{2})([A-Z0-9]+)$", event_ticker)
    return (m.group(1), m.group(2)) if m else None


def score_snapshot(snap: pd.DataFrame, games: pd.DataFrame, model: RatingModel, matcher: TeamMatcher) -> list[ScoredMarket]:
    """Score the latest snapshot rows of the three NCAA series against the schedule."""
    latest = snap[snap.ts == snap.ts.max()] if "ts" in snap else snap
    asof = pd.to_datetime(latest.ts.max(), utc=True) if "ts" in latest else pd.Timestamp.now(tz="UTC")
    # pre-game only: a market whose game has tipped already reflects the score, not a price
    upcoming = games[(~games.completed) & (games.tip_utc > asof + pd.Timedelta(minutes=5))]
    # team names per (date, codes) from the game series
    names: dict[tuple[str, str], list[str]] = {}
    for r in latest[latest.series == "KXNCAAMBGAME"].itertuples():
        ep = _event_parts(r.event_ticker)
        if ep and isinstance(r.yes_sub_title, str):
            names.setdefault(ep, [])
            if r.yes_sub_title not in names[ep]:
                names[ep].append(r.yes_sub_title)
    out: list[ScoredMarket] = []
    for r in latest.itertuples():
        ep = _event_parts(r.event_ticker)
        if ep is None or len(names.get(ep, [])) != 2:
            continue
        date_s, _ = ep
        day = pd.to_datetime(date_s, format="%y%b%d")
        ids = [matcher.match(n) for n in names[ep]]
        if None in ids:
            out.append(ScoredMarket(r.ticker, "?", r.title, str(day.date()), *names[ep], False, float("nan"), r.yes_bid_dollars, r.yes_ask_dollars, float("nan"), float("nan"), False, f"unmatched team {names[ep][ids.index(None)]}"))
            continue
        cand = upcoming[(upcoming.game_date.dt.normalize().between(day - pd.Timedelta(days=1), day + pd.Timedelta(days=1))) & (upcoming.home_id.isin(ids)) & (upcoming.away_id.isin(ids))]
        if cand.empty:
            out.append(ScoredMarket(r.ticker, "?", r.title, str(day.date()), *names[ep], False, float("nan"), r.yes_bid_dollars, r.yes_ask_dollars, float("nan"), float("nan"), False, "no schedule game"))
            continue
        gm = cand.iloc[0]
        home, away, neutral = gm.home_id, gm.away_id, bool(gm.neutral)
        p_home = model.p_home(home, away, neutral)
        title = r.title or ""
        if r.series == "KXNCAAMBGAME":
            team_id = matcher.match(r.yes_sub_title)
            p = p_home if team_id == home else 1 - p_home
            kind = "game"
        elif r.series == "KXNCAAMBSPREAD":
            m = re.match(r"^(.*?) wins by over ([\d.]+) points", title)
            if not m:
                continue
            team_id = matcher.match(m.group(1))
            line = float(m.group(2))
            p = model.p_cover(home, away, line, neutral) if team_id == home else model.p_cover(away, home, line, True) if neutral else 1 - model.p_cover(home, away, -line, neutral)
            kind = "spread"
        elif r.series == "KXNCAAMBTOTAL":
            m = re.search(r"Over ([\d.]+) points", title)
            if not m:
                continue
            p = model.p_over(home, away, float(m.group(1)))
            kind = "total"
        else:
            continue
        bid, ask = float(r.yes_bid_dollars), float(r.yes_ask_dollars)
        out.append(ScoredMarket(r.ticker, kind, title, str(gm.game_date.date()), gm.home_display_name, gm.away_display_name, neutral, float(p), bid, ask, float(p - ask), float((1 - p) - (1 - bid)), True))
    return out
