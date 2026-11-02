"""Read-only Kalshi public market data (no API key needed) and a snapshot recorder.

Series we care about (verified 2026-11-02): KXNCAAMBGAME (winner), KXNCAAMBSPREAD,
KXNCAAMBTOTAL, plus NBA/WNBA/EuroLeague equivalents. Prices are dollars per contract
(0-1). Each snapshot row is one market at one time; the ledger derives closing prices
from the last snapshot before tip-off.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_SERIES = ["KXNCAAMBGAME", "KXNCAAMBSPREAD", "KXNCAAMBTOTAL"]
FIELDS = [
    "ticker", "event_ticker", "title", "yes_sub_title", "status", "open_time", "close_time",
    "expected_expiration_time", "floor_strike", "cap_strike", "yes_bid_dollars", "yes_ask_dollars",
    "last_price_dollars", "volume", "volume_24h", "open_interest", "liquidity_dollars", "result",
]


def fetch_markets(series: str, status: str = "open", session: requests.Session | None = None, max_pages: int = 50) -> list[dict]:
    """All markets of a series (follows the cursor)."""
    s = session or requests.Session()
    out, cursor = [], None
    for _ in range(max_pages):
        params = {"series_ticker": series, "status": status, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = s.get(f"{BASE}/markets", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        d = r.json()
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor:
            break
        time.sleep(0.15)  # stay well under the public rate limit
    return out


def snapshot(series: list[str] | None = None, status: str = "open") -> pd.DataFrame:
    """One row per market with a UTC timestamp and the fields above."""
    series = series or DEFAULT_SERIES
    ts = datetime.now(timezone.utc)
    rows = []
    with requests.Session() as s:
        for ser in series:
            for m in fetch_markets(ser, status, s):
                row = {k: m.get(k) for k in FIELDS}
                row["series"] = ser
                row["ts"] = ts
                rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars", "liquidity_dollars", "floor_strike", "cap_strike"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if len(df):
        df["mid"] = (df.yes_bid_dollars + df.yes_ask_dollars) / 2
        df["spread"] = df.yes_ask_dollars - df.yes_bid_dollars
    return df


def append_snapshot(store: Path, series: list[str] | None = None) -> pd.DataFrame:
    """Append a snapshot to a dated parquet file under `store` and return it."""
    df = snapshot(series)
    store.mkdir(parents=True, exist_ok=True)
    f = store / f"snap_{datetime.now(timezone.utc):%Y%m%d}.parquet"
    if f.exists():
        df = pd.concat([pd.read_parquet(f), df], ignore_index=True)
    df.to_parquet(f, index=False)
    return df


def load_snapshots(store: Path) -> pd.DataFrame:
    files = sorted(Path(store).glob("snap_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
