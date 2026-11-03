"""Replay a past season against Kalshi's own prices.

The live paper ledger needs weeks to accumulate bets. Kalshi keeps every settled market
and its hourly candles, so the same question can be answered on last season today:

  1. match every settled KXNCAAMBGAME event to a schedule game (same matcher as live)
  2. take the quote Kalshi showed `hours_before` tip-off (last hourly candle with a two-sided book)
  3. compare the walk-forward model probability (made the morning of the game, no look-ahead)
     with the market, on the games' actual results
  4. simulate exactly the live betting policy (spread, price band, edge band, one bet per
     game, taker fee) and report P&L, ROI and closing-line value (entry vs the last pre-tip mid)

Three numbers matter, in this order: the blend coefficient (does the model carry information
the market lacks at all), the CLV of the policy's bets, and only then the P&L.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..kalshi.history import event_date, quote_before
from ..kalshi.ledger import taker_fee
from .kalshi_map import TeamMatcher


def match_events(events: pd.DataFrame, games: pd.DataFrame, matcher: TeamMatcher) -> pd.DataFrame:
    """One row per Kalshi event matched to a schedule game: event_ticker, game_id, and for the
    event's first market (alphabetical ticker) which team its YES side is."""
    g = games.copy()
    g["day"] = g.game_date.dt.normalize()
    rows = []
    for ev, grp in events.groupby("event_ticker"):
        names = [n for n in grp.yes_sub_title.dropna().unique() if isinstance(n, str)]
        day = event_date(ev)
        if len(names) != 2 or day is None:
            continue
        ids = [matcher.match(n) for n in names]
        if None in ids:
            rows.append(dict(event_ticker=ev, game_id=None, note=f"unmatched {names[ids.index(None)]}"))
            continue
        cand = g[g.day.between(day - pd.Timedelta(days=1), day + pd.Timedelta(days=1)) & g.home_id.isin(ids) & g.away_id.isin(ids)]
        if cand.empty:
            rows.append(dict(event_ticker=ev, game_id=None, note="no schedule game"))
            continue
        gm = cand.iloc[0]
        m = grp.sort_values("ticker").iloc[0]
        team = matcher.match(m.yes_sub_title)
        rows.append(
            dict(
                event_ticker=ev, game_id=gm.game_id, ticker=m.ticker, yes_team=m.yes_sub_title, yes_is_home=bool(team == gm.home_id),
                result=m.result, tip_utc=gm.tip_utc, event_day=day, volume=m.volume_fp, note="",
            )
        )
    return pd.DataFrame(rows)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def blend_test(y: np.ndarray, p_market: np.ndarray, p_model: np.ndarray, folds: int = 5, seed: int = 0) -> dict:
    """Logistic regression of outcome on logit(market) and logit(model).
    A positive, significant model coefficient means the model carries information the market
    price does not. Cross-validated log loss of market-only vs blend is the practical version."""
    from sklearn.linear_model import LogisticRegression

    X = np.column_stack([_logit(p_market), _logit(p_model)])
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    # standard errors from the Fisher information
    p = lr.predict_proba(X)[:, 1]
    Xd = np.column_stack([np.ones(len(y)), X])
    W = p * (1 - p)
    cov = np.linalg.pinv((Xd * W[:, None]).T @ Xd)
    se = np.sqrt(np.diag(cov))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    ll_m, ll_b = [], []
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        m1 = LogisticRegression(C=1e6, max_iter=1000).fit(X[tr, :1], y[tr])
        m2 = LogisticRegression(C=1e6, max_iter=1000).fit(X[tr], y[tr])
        ll_m.append(_logloss(y[te], m1.predict_proba(X[te, :1])[:, 1]))
        ll_b.append(_logloss(y[te], m2.predict_proba(X[te])[:, 1]))
    return dict(
        coef_market=float(lr.coef_[0, 0]), coef_model=float(lr.coef_[0, 1]),
        se_market=float(se[1]), se_model=float(se[2]), t_model=float(lr.coef_[0, 1] / se[2]),
        cv_logloss_market_only=float(np.mean(ll_m)), cv_logloss_blend=float(np.mean(ll_b)),
    )


def simulate_policy(df: pd.DataFrame, min_edge: float = 0.04, max_edge: float = 0.20, max_spread: float = 0.05, contracts: int = 20) -> pd.DataFrame:
    """The live `college predict --log-bets` rules applied to historical quotes.
    df needs: p_yes, bid, ask, y (1 if YES settled), close_mid (last pre-tip mid, for CLV)."""
    d = df.copy()
    ok = ((d.ask - d.bid) <= max_spread) & d.ask.between(0.05, 0.95)
    ey = d.p_yes - d.ask - np.array([taker_fee(a) for a in d.ask])
    en = (1 - d.p_yes) - (1 - d.bid) - np.array([taker_fee(1 - b) for b in d.bid])
    bet_yes = ok & ey.between(min_edge, max_edge)
    bet_no = ok & ~bet_yes & en.between(min_edge, max_edge)
    d["side"] = np.where(bet_yes, "yes", np.where(bet_no, "no", ""))
    d = d[d.side != ""].copy()
    d["price"] = np.where(d.side == "yes", d.ask, 1 - d.bid)
    d["edge"] = np.where(d.side == "yes", ey[d.index], en[d.index])
    d["fee"] = [taker_fee(p, contracts) for p in d.price]
    won = np.where(d.side == "yes", d.y == 1, d.y == 0)
    d["pnl"] = np.where(won, (1 - d.price) * contracts, -d.price * contracts) - d.fee
    close_side = np.where(d.side == "yes", d.close_mid, 1 - d.close_mid)
    d["clv"] = close_side - d.price
    d["contracts"] = contracts
    return d


def summarise_bets(b: pd.DataFrame) -> dict:
    if b.empty:
        return dict(bets=0)
    stake = float((b.price * b.contracts).sum())
    clv = b.clv.dropna()
    return dict(
        bets=int(len(b)), pnl=round(float(b.pnl.sum()), 2), stake=round(stake, 2), roi_on_stake=round(float(b.pnl.sum() / stake), 4),
        win_rate=round(float((b.pnl > 0).mean()), 3), mean_price=round(float(b.price.mean()), 3), mean_edge=round(float(b.edge.mean()), 4),
        mean_clv=round(float(clv.mean()), 4) if len(clv) else float("nan"),
        clv_positive_frac=round(float((clv > 0).mean()), 3) if len(clv) else float("nan"),
        clv_t_stat=round(float(clv.mean() / (clv.std(ddof=1) / np.sqrt(len(clv)))), 2) if len(clv) > 2 else float("nan"),
        pnl_t_stat=round(float(b.pnl.mean() / (b.pnl.std(ddof=1) / np.sqrt(len(b)))), 2) if len(b) > 2 else float("nan"),
        total_fees=round(float(b.fee.sum()), 2),
    )


def replay(pred: pd.DataFrame, games: pd.DataFrame, events: pd.DataFrame, candles: pd.DataFrame, matcher: TeamMatcher, hours_before: float = 1.0, max_quote_age_h: float = 24.0) -> tuple[pd.DataFrame, dict]:
    """Join predictions, schedule, Kalshi events and pre-tip quotes. Returns the per-game
    table (one row per matched game with a usable quote) and the summary dict."""
    mt = match_events(events, games, matcher)
    matched = mt[mt.game_id.notna()].copy()
    matched["game_id"] = matched.game_id.astype(int)
    d = matched.merge(pred[["game_id", "p_home", "date", "non_d1"]], on="game_id", how="inner")
    d["p_yes"] = np.where(d.yes_is_home, d.p_home, 1 - d.p_home)
    d["y"] = (d.result == "yes").astype(int)
    d = d[d.result.isin(["yes", "no"])]
    # quotes: entry `hours_before` tip, and the close (last candle before tip) for CLV
    tip = pd.to_datetime(d.set_index("ticker").tip_utc, utc=True)
    q = quote_before(candles, tip - pd.Timedelta(hours=hours_before))
    c = quote_before(candles, tip)[["ticker", "mid", "quote_age_h"]].rename(columns={"mid": "close_mid", "quote_age_h": "close_age_h"})
    d = d.merge(q, on="ticker", how="left").merge(c, on="ticker", how="left")
    n_matched, n_quoted = len(d), int(d.mid.notna().sum())
    d = d[d.mid.notna() & (d.quote_age_h <= max_quote_age_h)].copy()
    y = d.y.to_numpy(float)
    summary = dict(
        kalshi_events=int(events.event_ticker.nunique()), unmatched=int((mt.game_id.isna()).sum()), matched_with_prediction=n_matched,
        with_quote=n_quoted, with_fresh_quote=int(len(d)), hours_before=hours_before, median_quote_age_h=round(float(d.quote_age_h.median()), 2),
        median_spread=round(float((d.ask - d.bid).median()), 3),
        brier_market=round(float(np.mean((d.mid - y) ** 2)), 4), brier_model=round(float(np.mean((d.p_yes - y) ** 2)), 4),
        logloss_market=round(_logloss(y, d.mid), 4), logloss_model=round(_logloss(y, d.p_yes), 4),
        logit_corr=round(float(np.corrcoef(_logit(d.mid), _logit(d.p_yes))[0, 1]), 3),
        blend=blend_test(y, d.mid.to_numpy(), d.p_yes.to_numpy()) if len(d) > 50 else {},
    )
    bets = simulate_policy(d)
    summary["policy_live_rules"] = summarise_bets(bets)
    summary["by_min_edge"] = {f"{e:.2f}": summarise_bets(simulate_policy(d, min_edge=e)) for e in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)}
    if len(bets):
        bets["month"] = pd.to_datetime(bets.date).dt.strftime("%Y-%m")
        summary["policy_by_month"] = {m: summarise_bets(x) for m, x in bets.groupby("month")}
        summary["policy_d1_only"] = summarise_bets(bets[bets.non_d1 == 0])
    d["bet_side"] = d.ticker.map(bets.set_index("ticker").side) if len(bets) else ""
    d["bet_pnl"] = d.ticker.map(bets.set_index("ticker").pnl) if len(bets) else np.nan
    d["bet_clv"] = d.ticker.map(bets.set_index("ticker").clv) if len(bets) else np.nan
    return d, summary
