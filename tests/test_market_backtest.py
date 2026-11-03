import numpy as np
import pandas as pd

from hoops.college.market_backtest import blend_test, simulate_policy, summarise_bets
from hoops.kalshi.history import event_date, quote_before


def test_event_date():
    assert event_date("KXNCAAMBGAME-26FEB18UWGAEKY") == pd.Timestamp("2026-02-18")
    assert event_date("nonsense") is None


def test_quote_before_picks_last_two_sided_candle_before_cutoff():
    t0 = pd.Timestamp("2026-01-10 12:00", tz="UTC")
    c = pd.DataFrame(
        dict(
            ticker=["A"] * 4,
            end_ts=[t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=3)],
            bid_close=[0.40, 0.45, 0.0, 0.50], ask_close=[0.44, 0.47, 1.0, 0.52], price_close=[0.42, 0.46, 0.46, 0.51],
        )
    )
    q = quote_before(c, pd.Series({"A": t0 + pd.Timedelta(hours=2, minutes=30)}))
    assert len(q) == 1 and q.bid.iloc[0] == 0.45 and q.ask.iloc[0] == 0.47  # the one-sided 14:00 candle is skipped
    assert abs(q.quote_age_h.iloc[0] - 1.5) < 1e-9


def test_policy_matches_live_rules():
    d = pd.DataFrame(
        dict(
            ticker=list("abcd"),
            p_yes=[0.60, 0.30, 0.55, 0.90], bid=[0.50, 0.50, 0.50, 0.60], ask=[0.52, 0.52, 0.52, 0.62],
            y=[1, 0, 1, 1], close_mid=[0.55, 0.45, 0.51, 0.70],
        )
    )
    b = simulate_policy(d, min_edge=0.04, max_edge=0.20, max_spread=0.05, contracts=20)
    # a: yes edge 0.60-0.52-fee ~0.062 -> yes; b: no edge 0.70-0.50-fee ~0.18 -> no; c: edge < 0.04 -> skip; d: edge 0.26 > cap -> skip
    assert list(b.ticker) == ["a", "b"] and list(b.side) == ["yes", "no"]
    assert b.pnl.iloc[0] > 0 and b.pnl.iloc[1] > 0
    assert abs(b.clv.iloc[0] - (0.55 - 0.52)) < 1e-9 and abs(b.clv.iloc[1] - ((1 - 0.45) - 0.50)) < 1e-9
    s = summarise_bets(b)
    assert s["bets"] == 2 and s["pnl"] > 0


def test_blend_detects_informative_model():
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.normal(0, 1.2, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-truth))).astype(int)
    market = 1 / (1 + np.exp(-(truth + rng.normal(0, 0.8, n))))
    model_good = 1 / (1 + np.exp(-(truth + rng.normal(0, 0.8, n))))  # independent noisy view: adds information
    model_noise = 1 / (1 + np.exp(-rng.normal(0, 1.2, n)))
    good = blend_test(y, market, model_good)
    bad = blend_test(y, market, model_noise)
    assert good["t_model"] > 4 and good["cv_logloss_blend"] < good["cv_logloss_market_only"]
    assert abs(bad["t_model"]) < 3
