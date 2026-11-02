"""Paper-trading ledger and closing-line value (CLV).

A paper bet records: market ticker, side (yes/no), the price you would have paid,
your model probability, size, and time. Later, the *closing price* (the last mid
before tip-off) is attached from the snapshot store and CLV is computed as the
no-vig probability edge you had versus the market's final opinion:

    clv = close_prob - paid_price          (for a YES bet; sign-flipped for NO)

Positive CLV over a few hundred bets is the earliest reliable evidence of an edge;
P&L needs thousands of bets. Expected profit per contract after Kalshi's taker fee
0.07 * p * (1-p) is also tracked so fee drag is visible from day one.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def taker_fee(price: float, contracts: int = 1) -> float:
    """Kalshi taker fee in dollars: round-up(0.07 * C * P * (1-P)) to the cent."""
    cents = round(0.07 * contracts * price * (1 - price) * 100, 6)  # guard float noise before ceil
    return float(np.ceil(cents) / 100)


def break_even_prob(price: float, contracts: int = 100) -> float:
    """True probability needed to break even buying YES at `price` as taker."""
    fee = taker_fee(price, contracts) / contracts
    return price + fee


@dataclass
class PaperBet:
    ts: str
    ticker: str
    side: str  # yes | no
    price: float  # dollars paid per contract (for NO: 1 - yes_ask)
    model_prob: float  # model probability that YES settles (always for YES)
    contracts: int
    note: str = ""


def record_bet(ledger: Path, bet: PaperBet) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(bet)])
    if ledger.exists():
        df = pd.concat([pd.read_csv(ledger), df], ignore_index=True)
    df.to_csv(ledger, index=False)


def closing_prices(snapshots: pd.DataFrame, pre_tip_hours: float = 2.5) -> pd.DataFrame:
    """Last mid per market before (expected_expiration_time - pre_tip_hours) = approx tip-off.

    Kalshi markets do not carry the tip time; expected expiration is ~2.5-3 h after tip
    for basketball, so this is an approximation. Also returns the settled result when
    a later snapshot has one.
    """
    s = snapshots.dropna(subset=["mid"]).copy()
    s["ts"] = pd.to_datetime(s.ts, utc=True)
    s["exp"] = pd.to_datetime(s.expected_expiration_time, utc=True, errors="coerce")
    s["tip"] = s.exp - pd.Timedelta(hours=pre_tip_hours)
    pre = s[s.ts <= s.tip].sort_values("ts").groupby("ticker").tail(1)
    res = snapshots[snapshots.result.isin(["yes", "no"])].groupby("ticker").result.last()
    out = pre[["ticker", "ts", "mid", "yes_bid_dollars", "yes_ask_dollars", "tip"]].rename(columns={"ts": "close_ts", "mid": "close_mid"})
    out = out.merge(res.rename("result"), on="ticker", how="left")
    return out


def clv_report(ledger: Path, snapshots: pd.DataFrame, results: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join paper bets to closing prices; per-bet CLV, fee, and realised P&L when settled.
    `results` (ticker, result) from client.resolve_results overrides snapshot results,
    because settled markets stop appearing in open-market snapshots."""
    bets = pd.read_csv(ledger)
    close = closing_prices(snapshots)
    if results is not None and len(results):
        res = results[results.result.isin(["yes", "no"])][["ticker", "result"]].rename(columns={"result": "result_api"})
        close = close.merge(res, on="ticker", how="outer")
        close["result"] = close.result_api.combine_first(close.result) if "result" in close else close.result_api
        close = close.drop(columns=["result_api"])
    df = bets.merge(close, on="ticker", how="left")
    # a bet whose game has not tipped yet has no close: its 'close' is just the latest
    # price and must not count. Reference time = the latest snapshot (data-driven).
    now = pd.to_datetime(snapshots.ts, utc=True).max() if "ts" in snapshots and len(snapshots) else pd.Timestamp.now(tz="UTC")
    df["pending"] = df.close_mid.isna() | (pd.to_datetime(df.tip, utc=True, errors="coerce") > now)
    yes = df.side.str.lower() == "yes"
    close_prob = np.where(yes, df.close_mid, 1 - df.close_mid)
    df["close_prob"] = close_prob
    df["clv"] = df.close_prob - df.price
    df["fee"] = [taker_fee(p, c) for p, c in zip(df.price, df.contracts)]
    won = np.where(yes, df.result == "yes", df.result == "no")
    settled = df.result.isin(["yes", "no"])
    pnl = np.where(won, (1 - df.price) * df.contracts, -df.price * df.contracts) - df.fee
    df["pnl"] = np.where(settled, pnl, np.nan)
    df["model_edge_at_bet"] = np.where(yes, df.model_prob, 1 - df.model_prob) - df.price
    return df


def summarise(report: pd.DataFrame) -> dict:
    # CLV only counts once the game has tipped (the 'close' is final); earlier it is just
    # the current price, which moves until tip-off
    r = report.dropna(subset=["clv"])
    if "pending" in report:
        r = r[~report.pending.fillna(True)]
    settled = report.dropna(subset=["pnl"])
    out = {
        "bets": int(len(report)),
        "pending_tip": int(report.pending.sum()) if "pending" in report else 0,
        "with_close": int(len(r)),
        "mean_clv": float(r.clv.mean()) if len(r) else float("nan"),
        "clv_positive_frac": float((r.clv > 0).mean()) if len(r) else float("nan"),
        "clv_t_stat": float(r.clv.mean() / (r.clv.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 else float("nan"),
        "settled": int(len(settled)),
        "pnl": float(settled.pnl.sum()) if len(settled) else 0.0,
        "roi_on_stake": float(settled.pnl.sum() / (settled.price * settled.contracts).sum()) if len(settled) else float("nan"),
        "total_fees": float(report.fee.sum()),
    }
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
