"""Historical Kalshi market data for backtesting (public API, no key).

Settled markets vanish from `status=open` listings but stay on the API:
  /markets?series_ticker=..&status=settled            recent settlements (~2 months)
  /historical/markets?series_ticker=..                 archive (everything older)
  /series/{series}/markets/{ticker}/candlesticks       hourly bid/ask/price, recent markets
  /historical/markets/{ticker}/candlesticks            same for archived markets

So a whole past season can be replayed: for every game, the quote Kalshi showed an
hour before tip-off, and the settlement. Files under data/kalshi/history/:
  markets_<series>.parquet  one row per market (ticker, event, team, result, times)
  candles_<series>.parquet  hourly candles for one market per event (the other side is 1 - price)
Both are incremental: rerunning only fetches what is missing.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from .client import BASE

MARKET_FIELDS = [
    "ticker", "event_ticker", "title", "yes_sub_title", "status", "result", "open_time", "close_time",
    "expected_expiration_time", "settlement_ts", "volume_fp", "open_interest_fp", "last_price_dollars",
    "floor_strike", "cap_strike",
]


def _get(s: requests.Session, path: str, **params) -> requests.Response:
    r = None
    for attempt in range(8):
        r = s.get(f"{BASE}{path}", params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1.5 * (attempt + 1))
            continue
        return r
    return r


def event_date(event_ticker: str) -> pd.Timestamp | None:
    m = re.match(r"^KX\w+?-(\d{2}[A-Z]{3}\d{2})[A-Z0-9]+$", event_ticker)
    return pd.to_datetime(m.group(1), format="%y%b%d") if m else None


def _paginate(s: requests.Session, path: str, params: dict, max_pages: int, log) -> list[dict]:
    out, cursor = [], None
    for page in range(max_pages):
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        r = _get(s, path, **p)
        r.raise_for_status()
        d = r.json()
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if page % 10 == 9:
            log(f"  {path} page {page + 1}: {len(out)} markets")
        if not cursor:
            break
        time.sleep(0.15)
    return out


def fetch_settled_markets(series: str, since: str, session: requests.Session | None = None, log=print) -> pd.DataFrame:
    """Recent settled + archived markets of `series` whose event date >= `since` (YYYY-MM-DD)."""
    s = session or requests.Session()
    rows = _paginate(s, "/markets", dict(series_ticker=series, status="settled", limit=1000), 50, log)
    rows += _paginate(s, "/historical/markets", dict(series_ticker=series, limit=1000), 200, log)
    df = pd.DataFrame([{k: m.get(k) for k in MARKET_FIELDS} for m in rows]).drop_duplicates("ticker")
    df["event_date"] = df.event_ticker.map(event_date)
    df = df[df.event_date >= pd.Timestamp(since)].copy()
    df["series"] = series
    for c in ("volume_fp", "open_interest_fp", "last_price_dollars", "floor_strike", "cap_strike"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["event_date", "ticker"]).reset_index(drop=True)


def _num(d: dict | None, key: str):
    """Candle sub-dict value: recent markets use `<key>_dollars`, archived ones plain `<key>`."""
    if not d:
        return None
    v = d.get(f"{key}_dollars", d.get(key))
    return None if v is None else float(v)


def fetch_candles(series: str, ticker: str, start: datetime, end: datetime, interval_min: int = 60, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    params = dict(start_ts=int(start.timestamp()), end_ts=int(end.timestamp()), period_interval=interval_min)
    r = _get(s, f"/series/{series}/markets/{ticker}/candlesticks", **params)
    if r.status_code == 404:
        r = _get(s, f"/historical/markets/{ticker}/candlesticks", **params)
    r.raise_for_status()
    rows = []
    for c in r.json().get("candlesticks", []):
        p, b, a = c.get("price"), c.get("yes_bid"), c.get("yes_ask")
        rows.append(
            dict(
                ticker=ticker,
                end_ts=pd.to_datetime(int(c["end_period_ts"]), unit="s", utc=True),
                price_close=_num(p, "close"), price_mean=_num(p, "mean"), price_prev=_num(p, "previous"),
                bid_close=_num(b, "close"), ask_close=_num(a, "close"),
                volume=float(c.get("volume_fp", c.get("volume")) or 0), open_interest=float(c.get("open_interest_fp", c.get("open_interest")) or 0),
            )
        )
    return pd.DataFrame(rows)


def build_history(store: Path, series: str = "KXNCAAMBGAME", since: str = "2025-11-01", refresh_markets: bool = False, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (incrementally) settled markets and one market's hourly candles per event."""
    hist = Path(store) / "history"
    hist.mkdir(parents=True, exist_ok=True)
    mf, cf = hist / f"markets_{series}.parquet", hist / f"candles_{series}.parquet"
    with requests.Session() as s:
        if mf.exists() and not refresh_markets:
            markets = pd.read_parquet(mf)
        else:
            log(f"fetching settled {series} markets since {since}")
            markets = fetch_settled_markets(series, since, s, log=log)
            markets.to_parquet(mf, index=False)
        log(f"{len(markets)} markets in {markets.event_ticker.nunique()} events, {markets.event_date.min().date()} .. {markets.event_date.max().date()}")
        # one market per event (the first side); the other is the complement
        first = markets.sort_values("ticker").groupby("event_ticker").head(1)
        candles = pd.read_parquet(cf) if cf.exists() else pd.DataFrame(columns=["ticker"])
        done = set(candles.ticker)
        todo = first[~first.ticker.isin(done)]
        log(f"candles: {len(done)} markets done, {len(todo)} to fetch")
        buf, t0 = [], time.time()
        for i, m in enumerate(todo.itertuples(), 1):
            start = pd.to_datetime(m.open_time, utc=True) - pd.Timedelta(hours=1)
            end = pd.to_datetime(m.close_time, utc=True) + pd.Timedelta(hours=1)
            try:
                c = fetch_candles(series, m.ticker, start.to_pydatetime(), end.to_pydatetime(), session=s)
                buf.append(c if len(c) else pd.DataFrame([dict(ticker=m.ticker)]))  # remember empty ones too
            except requests.HTTPError as e:
                log(f"  {m.ticker}: {e}")
            time.sleep(0.1)
            if i % 250 == 0 or i == len(todo):
                new = pd.concat(buf, ignore_index=True) if buf else pd.DataFrame()
                candles = pd.concat([candles, new], ignore_index=True) if len(new) else candles
                candles.to_parquet(cf, index=False)
                buf = []
                rate = i / max(time.time() - t0, 1e-6)
                log(f"  {i}/{len(todo)} markets, {len(candles)} candles, {rate:.1f}/s, ~{(len(todo) - i) / max(rate, 1e-6) / 60:.0f} min left")
    return markets, candles


def quote_before(candles: pd.DataFrame, cutoff: pd.Series) -> pd.DataFrame:
    """Per ticker: the last hourly candle ending at or before `cutoff[ticker]` with a two-sided quote.
    Returns ticker, quote_ts, bid, ask, mid, last_price, quote_age_h (hours between the candle and the cutoff)."""
    c = candles.dropna(subset=["bid_close", "ask_close"]).copy()
    c = c[(c.bid_close > 0) & (c.ask_close < 1)]
    c["cutoff"] = c.ticker.map(cutoff)
    c = c[c.end_ts <= c.cutoff].sort_values("end_ts").groupby("ticker").tail(1)
    out = c[["ticker", "end_ts", "bid_close", "ask_close", "price_close", "cutoff"]].rename(columns={"end_ts": "quote_ts", "bid_close": "bid", "ask_close": "ask", "price_close": "last_price"})
    out["mid"] = (out.bid + out.ask) / 2
    out["quote_age_h"] = (out.cutoff - out.quote_ts).dt.total_seconds() / 3600
    return out.drop(columns=["cutoff"]).reset_index(drop=True)
