# hoops — basketball broadcast tracking → court coordinates → win probability

Personal project: take NBA-style broadcast footage, track players and the ball
with current open-source state-of-the-art components, convert to court
coordinates, and feed a win-probability model. Every modelling choice (tracker,
motion model, calibration method) is selected by measured benchmarks on
basketball data, not by paper claims.

Plan and research notes: see the plan file referenced in the session; the
verified component choices (2026-10-28) are summarised below.

## Setup

```bash
uv sync --extra dev              # Python 3.12, torch on MPS (Mac) or CUDA (A100)
uv run hoops info                # device + versions
uv run pytest                    # unit tests (motion models, court calibration, trackers, metrics)
uv run python scripts/download_weights.py        # Deep-EIoU ReID, WASB ball, ultralytics ReID
uv run python scripts/download_datasets.py sportsmot   # prints the registration steps
```

## Pipeline

| Stage | Module | Status |
|---|---|---|
| Video + scene cuts | `hoops/ingest` (OpenCV, PySceneDetect) | done |
| Detection | `hoops/detect/rfdetr_detector.py` (RF-DETR, Apache 2.0; COCO now, basketball fine-tune on the A100) | COCO works on MPS (~20 fps @720p) |
| Motion models | `hoops/motion` Kalman CV / CA, IMM, particle filter (tested vs filterpy) | done |
| Trackers | `hoops/track/ours.py` + adapters for `trackers`, `boxmot`, `ultralytics` (11 candidates behind one registry) | done |
| ReID | `hoops/reid/boxmot_reid.py` (OSNet; Deep-EIoU sports weights via path) | done |
| Court calibration | `hoops/court`: geometry (41 landmarks, 5 league specs), RANSAC homography with line-support candidate selection and refinement, PTZ camera + Kalman + floor flow, quality gate; landmark model `weights/court/court_bcd33_s1280_best.pt` (YOLO-pose small, 33 pts) | external test median 0.12 m, 77% of frames accepted (BENCHMARKS.md) |
| Metrics / benchmark | `hoops/eval` HOTA (matches TrackEval), IDF1, MOTA; `hoops bench` on SportsMOT basketball val | 12-tracker sweep done, ours leads at 0.476 HOTA with a generic detector |
| Team split | `hoops/team/siglip_cluster.py` (SigLIP → UMAP → KMeans) | code written, untested on footage |
| Tracking features | `hoops/features/tracking.py` (shared `TrackingFrame` schema: spacing, nearest defender, pace, possession) | done |
| Win probability | `hoops/wp` game-state baseline (LightGBM + logistic on 1996–2025 play-by-play); tracking-augmented model next | baseline trained, see BENCHMARKS.md |
| Other leagues | `--court ncaa_men / ncaa_men_pre2019 / ncaa_women / fiba` (same landmark ids, different coordinates); line-alignment refinement + plausibility gate | works coarsely on NCAA/FIBA clips without new labels (BENCHMARKS.md) |
| Kalshi paper trading | `hoops/kalshi`: public-API snapshot recorder (no key), paper-bet ledger, closing-line value and fee maths | recording NCAA game/spread/total markets every 10 min |
| College pre-game model | `hoops/college`: ridge margin ratings on free sportsdataverse scores, walk-forward backtest, Kalshi market matcher and scorer, replay against Kalshi's historical prices | Brier 0.166 on 2025-26 walk-forward, but no information beyond Kalshi's closing price (replay: -10% ROI over 1,199 simulated bets); paper bets logging since 2 Nov 2026 |

## What needs you

1. **NBA broadcast clips** for `hoops track` / `hoops calibrate` end to end. SportsMOT's basketball clips are NCAA and FIBA footage, which the NBA court model and geometry do not fit.
2. **A100 on Thunder Compute** for the RF-DETR basketball fine-tune (`scripts/train_detector.py`), the biggest remaining lever on tracking HOTA.
3. **SportVU 2015-16** (3.7 GB): `scripts/download_datasets.py sportvu` when you want the tracking-augmented win-probability model.

Datasets already in place: SportsMOT val (official Hugging Face mirror, `scripts/prepare_sportsmot.py`),
Roboflow basketball detection (464 train images, 10 classes), the 33-point court set
(`bcd2_v21`) and the 80-image court set kept as external test, play-by-play 1996-2025.
Long jobs: `bash scripts/resume_jobs.sh` restarts training / benchmark after a shutdown.

## Commands

```bash
uv run hoops track game.mp4 --tracker hoops:imm --reid boxmot:osnet-x0-25-msmt17
uv run hoops cuts game.mp4
uv run hoops calibrate game.mp4 --landmarks manual:points.json --tracks out/game_tracks.parquet
uv run hoops wp-train --train-seasons 2010-2022 --val-seasons 2023
uv run hoops calibrate nba_clip.mp4 --landmarks yolo:weights/court/court_bcd33_s1280_best.pt@bcd33
uv run hoops calibrate college.mp4 --landmarks yolo:weights/court/court_bcd33_n960_best.pt@bcd33 --court ncaa_men   # n960 transfers better off-NBA
uv run hoops kalshi snapshot --loop-minutes 10        # record NCAA market prices (no API key)
uv run hoops kalshi bet KXNCAAMBGAME-26NOV05TOWSUTAH-UTAH yes 0.68 0.74 --contracts 20
uv run hoops kalshi clv                                # closing-line value of the paper ledger
uv run hoops college backtest --season 2026           # walk-forward rating model, calibrated on the previous season
uv run hoops college predict --log-bets               # score today's Kalshi NCAA markets, log paper bets (liquid game markets)
uv run hoops kalshi history --since 2025-11-01        # download a past season's settled markets + hourly candles (public archive)
uv run hoops college market-backtest --hours-before 1 # replay the season against Kalshi's prices: market vs model, blend test, policy P&L/CLV
bash scripts/college_daily.sh                         # the whole daily loop: refresh scores, snapshot, predict, CLV
uv run hoops bench --data data/sportsmot --trackers hoops:kalman_cv,hoops:imm,trackers:bytetrack,ultralytics:tracktrack
```

## Licences

`rfdetr`, `trackers`, `supervision`: Apache 2.0. `ultralytics`, `boxmot`: AGPL-3.0
(fine for personal use; benchmarked as alternatives). WASB weights: research only.
SportsMOT: no redistribution.
