"""`hoops` command line.

  hoops info                       # device + versions
  hoops cuts VIDEO                 # scene cuts
  hoops track VIDEO [--tracker hoops:imm --detector rfdetr:medium --reid boxmot:osnet-x0-25-msmt17]
  hoops bench --data data/sportsmot [--trackers hoops:kalman_cv,trackers:bytetrack ...]
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def info():
    from hoops.device import device_summary

    rprint(f"[bold]device[/]: {device_summary()}")
    for mod in ("rfdetr", "ultralytics", "boxmot", "supervision", "scenedetect"):
        try:
            m = __import__(mod)
            rprint(f"{mod:12s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # pragma: no cover
            rprint(f"{mod:12s} [red]{e}[/]")


@app.command()
def cuts(video: Path, threshold: float = 27.0):
    from hoops.ingest.cuts import detect_cuts, segments_from_cuts
    from hoops.ingest.video import probe

    info = probe(video)
    c = detect_cuts(video, threshold)
    segs = segments_from_cuts(c, info.n_frames)
    rprint(f"{len(segs)} segments over {info.n_frames} frames @ {info.fps:.1f} fps")
    for s in segs:
        rprint(f"  seg {s.id:3d}: frames {s.start:6d}-{s.end:6d}  ({(s.end - s.start) / info.fps:6.1f}s)")


@app.command()
def track(
    video: Path,
    out: Path = Path("out"),
    detector: str = "rfdetr:medium",
    tracker: str = "hoops:kalman_cv",
    reid: str = "none",
    conf: float = 0.3,
    device: str = typer.Option(None, help="cuda | mps | cpu (default: auto)"),
    no_cuts: bool = False,
    no_overlay: bool = False,
    start: int = 0,
    end: int = typer.Option(None),
    stride: int = 1,
):
    from hoops.pipeline import track_video
    from hoops.registry import make_detector, make_embedder, make_tracker

    det = make_detector(detector, device=device, conf=conf)
    emb = make_embedder(reid, device=device)
    trk = make_tracker(tracker, embedder=emb, device=device)
    run = track_video(video, det, trk, out, not no_cuts, not no_overlay, start, end, stride)
    rprint(f"[green]done[/] {run.n_frames} frames at {run.fps_processed:.1f} fps -> {run.tracks_parquet}")
    if run.overlay_mp4:
        rprint(f"overlay: {run.overlay_mp4}")
    rprint({k: round(v, 2) for k, v in run.timings.items()})


@app.command()
def calibrate(
    video: Path,
    landmarks: str = typer.Option(..., help="manual:points.json | yolo:weights.pt | rfdetr:weights.pth"),
    out: Path = Path("out"),
    tracks: Path = typer.Option(None, help="tracks parquet from `hoops track` to get court coordinates"),
    court: str = typer.Option("nba", help="nba | ncaa_men | ncaa_men_pre2019 | ncaa_women | fiba"),
    device: str = typer.Option(None),
    no_overlay: bool = False,
    no_cuts: bool = False,
    start: int = 0,
    end: int = typer.Option(None),
):
    from hoops.calibrate_pipeline import calibrate_video
    from hoops.court.geometry import set_court
    from hoops.court.keypoints import make_landmark_source

    set_court(court)
    src = make_landmark_source(landmarks, device=device)
    run = calibrate_video(video, src, out, tracks, not no_overlay, not no_cuts, start, end)
    n = max(run.n_frames, 1)
    rprint(f"[green]done[/] {run.n_frames} frames: accepted {run.accepted / n:.0%}, uncertain {run.uncertain / n:.0%}, rejected {run.rejected / n:.0%}")
    rprint(f"calibration: {run.calib_npz}")
    if run.court_parquet:
        rprint(f"court coordinates: {run.court_parquet}")
    if run.overlay_mp4:
        rprint(f"overlay: {run.overlay_mp4}")


@app.command()
def bench(
    data: Path = Path("data/sportsmot"),
    split: str = "val",
    trackers: str = "hoops:kalman_cv,hoops:imm,trackers:bytetrack,trackers:botsort",
    detector: str = "rfdetr:medium",
    weights: Path = typer.Option(None, help="fine-tuned detector checkpoint (cached separately)"),
    reid: str = "none",
    conf: float = 0.3,
    device: str = typer.Option(None),
    max_seqs: int = typer.Option(None, help="limit sequences for a quick run"),
    out: Path = Path("out/bench"),
):
    from hoops.eval.bench import run_bench

    results = run_bench(data, split, trackers.split(","), detector, reid, conf, device, max_seqs, out, weights=weights)
    t = Table(title=f"SportsMOT {split} basketball ({detector}{' + ' + weights.stem if weights else ''}, reid={reid})")
    for col in ("tracker", "HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "fps"):
        t.add_column(col)
    for r in results:
        t.add_row(r["tracker"], *(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "fps")))
    rprint(t)


@app.command(name="wp-train")
def wp_train(
    train_seasons: str = "2010-2022",
    val_seasons: str = "2023",
    out: Path = Path("out/wp/baseline.joblib"),
    sample_frac: float = 1.0,
):
    """Train the game-state win-probability baseline (LightGBM + logistic) on play-by-play."""
    import json

    from hoops.wp.baseline import train

    def parse(s: str) -> list[int]:
        a, _, b = s.partition("-")
        return list(range(int(a), int(b or a) + 1))

    models, report = train(parse(train_seasons), parse(val_seasons), sample_frac)
    models.save(out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    t = Table(title=f"win probability, validation seasons {val_seasons}")
    for col in ("model", "log_loss", "brier", "ece", "ll_q1", "ll_q4", "ll_last2min"):
        t.add_column(col)
    for name in ("gbm", "logit", "naive_home"):
        r = report[name]
        t.add_row(name, f"{r['log_loss']:.4f}", f"{r['brier']:.4f}", f"{r['ece']:.4f}", f"{r.get('log_loss_q1', float('nan')):.4f}", f"{r.get('log_loss_q4', float('nan')):.4f}", f"{r.get('log_loss_last2min', float('nan')):.4f}")
    rprint(t)
    rprint(f"saved {out}")


kalshi_app = typer.Typer(no_args_is_help=True, help="Kalshi public market snapshots and the paper-trading ledger")
app.add_typer(kalshi_app, name="kalshi")


@kalshi_app.command("snapshot")
def kalshi_snapshot(
    series: str = "KXNCAAMBGAME,KXNCAAMBSPREAD,KXNCAAMBTOTAL",
    store: Path = Path("data/kalshi"),
    loop_minutes: float = typer.Option(0, help="repeat every N minutes (0 = once)"),
):
    """Record bid/ask/mid for every open market of the given series (no API key needed)."""
    import time

    from hoops.kalshi.client import append_snapshot

    while True:
        df = append_snapshot(store, series.split(","))
        latest = df[df.ts == df.ts.max()]
        rprint(f"{latest.ts.iloc[0]:%Y-%m-%d %H:%M} UTC: {len(latest)} markets, median spread {latest.spread.median():.3f}, stored in {store}")
        if loop_minutes <= 0:
            break
        time.sleep(loop_minutes * 60)


@kalshi_app.command("bet")
def kalshi_bet(ticker: str, side: str, price: float, model_prob: float, contracts: int = 10, note: str = "", ledger: Path = Path("data/kalshi/paper_ledger.csv")):
    """Log a paper bet (no money): what you would have bought, at what price, and what your model said."""
    from hoops.kalshi.ledger import PaperBet, break_even_prob, now_iso, record_bet

    record_bet(ledger, PaperBet(now_iso(), ticker, side, price, model_prob, contracts, note))
    rprint(f"logged {side} {ticker} @ {price:.2f} x{contracts}; break-even true prob {break_even_prob(price, contracts):.3f}")


@kalshi_app.command("clv")
def kalshi_clv(ledger: Path = Path("data/kalshi/paper_ledger.csv"), store: Path = Path("data/kalshi")):
    """Closing-line value and P&L of the paper ledger against recorded snapshots."""
    import pandas as pd

    from hoops.kalshi.client import load_snapshots, resolve_results
    from hoops.kalshi.ledger import clv_report, summarise

    snaps = load_snapshots(store)
    if snaps.empty or not ledger.exists():
        rprint("[yellow]need both snapshots and a ledger[/]")
        raise typer.Exit(1)
    results = resolve_results(pd.read_csv(ledger).ticker.tolist(), store)
    rep = clv_report(ledger, snaps, results)
    rprint(summarise(rep))
    rprint(rep[["ts", "ticker", "side", "price", "model_prob", "close_prob", "clv", "pnl"]].tail(15).to_string())


@kalshi_app.command("history")
def kalshi_history(series: str = "KXNCAAMBGAME", since: str = "2025-11-01", store: Path = Path("data/kalshi"), refresh_markets: bool = False):
    """Download settled markets and hourly candles of a past season (incremental; public API)."""
    from hoops.kalshi.history import build_history

    events, candles = build_history(store, series, since, refresh_markets=refresh_markets, log=rprint)
    rprint(f"{len(events)} markets, {len(candles)} candles under {store / 'history'}")


college_app = typer.Typer(no_args_is_help=True, help="Pre-game college model: ratings, backtest, scoring Kalshi markets")
app.add_typer(college_app, name="college")


@college_app.command("backtest")
def college_backtest(season: int = 2026, seasons_loaded: str = "2024-2027", out: Path = Path("out/college")):
    """Walk-forward backtest of the rating model on one season (prior = previous season)."""
    import json

    from hoops.college.backtest import calibrate, summarise, walk_forward
    from hoops.college.data import load_games

    a, _, b = seasons_loaded.partition("-")
    games = load_games(list(range(int(a), int(b or a) + 1)))
    out.mkdir(parents=True, exist_ok=True)
    # calibration fitted on the PREVIOUS season's walk-forward (no look-ahead into `season`)
    prev = walk_forward(games, season - 1)
    scale, sigma = calibrate(prev)
    rprint(f"calibration from season {season - 1}: margin scale {scale:.2f}, sigma {sigma:.1f} (raw: {summarise(prev)['brier']:.4f} brier)")
    (out / "calibration.json").write_text(json.dumps({"from_season": season - 1, "scale": scale, "sigma": sigma}))
    pred = walk_forward(games, season, scale=scale, sigma=sigma)
    s = summarise(pred)
    pred.to_parquet(out / f"backtest_{season}.parquet", index=False)
    (out / f"backtest_{season}.json").write_text(json.dumps(s, indent=2, default=str))
    rprint({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k not in ("by_month", "calibration")})
    rprint("by month:", s["by_month"])
    rprint("calibration (pred, observed, n):", s["calibration"])


@college_app.command("predict")
def college_predict(
    store: Path = Path("data/kalshi"),
    season: int = 2027,
    min_edge: float = typer.Option(0.04, help="log paper bets when model edge after the taker fee exceeds this"),
    max_edge: float = typer.Option(0.20, help="larger 'edges' are almost always a mismatch or a stale quote: skipped"),
    max_spread: float = typer.Option(0.05, help="only quote-tight markets (ask - bid) are bettable"),
    kinds: str = typer.Option("game", help="comma list of game,spread,total to consider for paper bets"),
    log_bets: bool = False,
    contracts: int = 20,
    ledger: Path = Path("data/kalshi/paper_ledger.csv"),
    refresh: bool = True,
):
    """Score the latest Kalshi NCAA snapshot with today's ratings; optionally log paper bets."""
    from hoops.college.data import load_games, team_table
    from hoops.college.kalshi_map import TeamMatcher, score_snapshot
    from hoops.college.ratings import fit_ratings
    from hoops.kalshi.client import load_snapshots
    from hoops.kalshi.ledger import PaperBet, now_iso, record_bet, taker_fee

    import json

    from hoops.college.data import non_d1_teams

    games = load_games([season - 1, season], refresh_current=refresh, current_season=season)
    nd1 = non_d1_teams(games)
    prior = fit_ratings(games[games.season == season - 1], non_d1=nd1)
    model = fit_ratings(games[games.season == season], prior=prior, non_d1=nd1)
    cal = Path("out/college/calibration.json")
    if cal.exists():
        c = json.loads(cal.read_text())
        model.scale, model.sigma = c["scale"], c["sigma"]
    rprint(f"ratings: {model.n_games} games this season, margin scale {model.scale:.2f}, sigma {model.sigma:.1f}, HFA {model.hfa:.2f}")
    snaps = load_snapshots(store)
    if snaps.empty:
        rprint("[yellow]no snapshots; run `hoops kalshi snapshot` first[/]")
        raise typer.Exit(1)
    scored = score_snapshot(snaps, games, model, TeamMatcher(team_table(games)))
    matched = [s for s in scored if s.matched]
    rprint(f"{len(scored)} markets in the latest snapshot, {len(matched)} matched to schedule games")
    t = Table(title="largest model edges (after taker fee)")
    for c in ("kind", "market", "date", "model P(yes)", "bid", "ask", "edge yes", "edge no"):
        t.add_column(c)
    wanted = set(kinds.split(","))
    rows = []
    for s in matched:
        if s.kind not in wanted or (s.yes_ask - s.yes_bid) > max_spread or not (0.05 <= s.yes_ask <= 0.95):
            continue
        fee_yes, fee_no = taker_fee(s.yes_ask), taker_fee(1 - s.yes_bid)
        rows.append((s, s.edge_yes - fee_yes, s.edge_no - fee_no))
    rows.sort(key=lambda x: -max(x[1], x[2]))
    rprint(f"{len(rows)} liquid {kinds} markets (spread <= {max_spread:.2f}, price 0.05-0.95)")
    for s, ey, en in rows[:20]:
        t.add_row(s.kind, s.title[:40], s.game_date, f"{s.yes_prob:.3f}", f"{s.yes_bid:.2f}", f"{s.yes_ask:.2f}", f"{ey:+.3f}", f"{en:+.3f}")
    rprint(t)
    if log_bets:
        import pandas as pd

        already = set(pd.read_csv(ledger).ticker.str.rsplit("-", n=1).str[0]) if ledger.exists() else set()
        n = 0
        for s, ey, en in rows:
            event = s.ticker.rsplit("-", 1)[0]
            if event in already:
                continue  # one paper bet per game
            if min_edge <= ey <= max_edge:
                record_bet(ledger, PaperBet(now_iso(), s.ticker, "yes", s.yes_ask, s.yes_prob, contracts, f"{s.kind} auto"))
            elif min_edge <= en <= max_edge:
                record_bet(ledger, PaperBet(now_iso(), s.ticker, "no", 1 - s.yes_bid, s.yes_prob, contracts, f"{s.kind} auto"))
            else:
                continue
            already.add(event)
            n += 1
        rprint(f"logged {n} paper bets to {ledger}")


@college_app.command("market-backtest")
def college_market_backtest(
    season: int = 2026,
    hours_before: float = typer.Option(1.0, help="use the Kalshi quote this many hours before tip-off"),
    max_quote_age_h: float = typer.Option(24.0, help="drop games whose last quote is older than this"),
    store: Path = Path("data/kalshi"),
    out: Path = Path("out/college"),
    download: bool = typer.Option(True, help="fetch missing Kalshi history first"),
):
    """Replay a season's walk-forward predictions against Kalshi's historical prices:
    market vs model accuracy, blend test, and the live betting policy's P&L and CLV."""
    import json

    import pandas as pd

    from hoops.college.data import load_games, team_table
    from hoops.college.kalshi_map import TeamMatcher
    from hoops.college.market_backtest import replay
    from hoops.kalshi.history import build_history

    pf = out / f"backtest_{season}.parquet"
    if not pf.exists():
        rprint(f"[yellow]no walk-forward predictions at {pf}; run `hoops college backtest --season {season}` first[/]")
        raise typer.Exit(1)
    pred = pd.read_parquet(pf)
    games = load_games([season - 1, season])
    since = f"{season - 1}-11-01"
    if download:
        events, candles = build_history(store, "KXNCAAMBGAME", since, log=rprint)
    else:
        events = pd.read_parquet(store / "history" / "markets_KXNCAAMBGAME.parquet")
        candles = pd.read_parquet(store / "history" / "candles_KXNCAAMBGAME.parquet")
    table, summary = replay(pred, games[games.season == season], events, candles, TeamMatcher(team_table(games)), hours_before, max_quote_age_h)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{season}_h{hours_before:g}"
    table.to_parquet(out / f"market_backtest_{tag}.parquet", index=False)
    (out / f"market_backtest_{tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    for k, v in summary.items():
        if k not in ("by_min_edge", "policy_by_month"):
            rprint(f"{k}: {v}")
    rprint("by min edge:")
    for e, v in summary["by_min_edge"].items():
        rprint(f"  {e}: {v}")
    if "policy_by_month" in summary:
        rprint("live policy by month:")
        for m, v in summary["policy_by_month"].items():
            rprint(f"  {m}: bets {v['bets']} pnl {v['pnl']} roi {v['roi_on_stake']} clv {v['mean_clv']}")


if __name__ == "__main__":
    app()
