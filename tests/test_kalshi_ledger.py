import pandas as pd

from hoops.kalshi.ledger import PaperBet, break_even_prob, closing_prices, clv_report, record_bet, summarise, taker_fee


def test_fee_formula_matches_known_values():
    assert taker_fee(0.5, 100) == 1.75
    assert taker_fee(0.9, 100) == 0.63
    assert taker_fee(0.5, 1) == 0.02  # rounds up to the cent
    assert abs(break_even_prob(0.5, 100) - 0.5175) < 1e-9


def test_clv_and_pnl(tmp_path):
    ledger = tmp_path / "ledger.csv"
    record_bet(ledger, PaperBet("2026-11-05T20:00:00+00:00", "KX-A", "yes", 0.55, 0.62, 100))
    record_bet(ledger, PaperBet("2026-11-05T20:00:00+00:00", "KX-B", "no", 0.40, 0.45, 50))  # NO at 0.40 => yes_ask 0.60
    snaps = pd.DataFrame(
        [
            # KX-A: pre-tip mids 0.55 then 0.60 (close), post-tip 0.9, settled yes
            dict(ticker="KX-A", ts="2026-11-05T20:00:00Z", expected_expiration_time="2026-11-06T04:00:00Z", mid=0.55, yes_bid_dollars=0.54, yes_ask_dollars=0.56, result=""),
            dict(ticker="KX-A", ts="2026-11-06T01:00:00Z", expected_expiration_time="2026-11-06T04:00:00Z", mid=0.60, yes_bid_dollars=0.59, yes_ask_dollars=0.61, result=""),
            dict(ticker="KX-A", ts="2026-11-06T03:00:00Z", expected_expiration_time="2026-11-06T04:00:00Z", mid=0.90, yes_bid_dollars=0.89, yes_ask_dollars=0.91, result=""),
            dict(ticker="KX-A", ts="2026-11-06T05:00:00Z", expected_expiration_time="2026-11-06T04:00:00Z", mid=1.0, yes_bid_dollars=1.0, yes_ask_dollars=1.0, result="yes"),
            # KX-B: close mid 0.65 (yes), settled yes => our NO loses
            dict(ticker="KX-B", ts="2026-11-06T01:00:00Z", expected_expiration_time="2026-11-06T04:00:00Z", mid=0.65, yes_bid_dollars=0.64, yes_ask_dollars=0.66, result=""),
            dict(ticker="KX-B", ts="2026-11-06T05:00:00Z", expected_expiration_time="2026-11-06T04:00:00Z", mid=1.0, yes_bid_dollars=1.0, yes_ask_dollars=1.0, result="yes"),
        ]
    )
    close = closing_prices(snaps)
    assert abs(float(close.set_index("ticker").loc["KX-A", "close_mid"]) - 0.60) < 1e-9
    rep = clv_report(ledger, snaps)
    a, b = rep.set_index("ticker").loc["KX-A"], rep.set_index("ticker").loc["KX-B"]
    assert abs(a.clv - 0.05) < 1e-9  # bought 0.55, closed 0.60
    assert abs(b.clv - ((1 - 0.65) - 0.40)) < 1e-9  # NO bought at 0.40, close NO-prob 0.35
    assert abs(a.pnl - ((1 - 0.55) * 100 - taker_fee(0.55, 100))) < 1e-9
    assert abs(b.pnl - (-0.40 * 50 - taker_fee(0.40, 50))) < 1e-9
    s = summarise(rep)
    assert s["bets"] == 2 and s["settled"] == 2 and s["clv_positive_frac"] == 0.5
