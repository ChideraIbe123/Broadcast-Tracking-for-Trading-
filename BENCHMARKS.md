# Benchmarks

All numbers are measured, not quoted. Re-run with the commands shown.

## Win probability, game-state baseline (play-by-play only)

`uv run hoops wp-train --train-seasons 2015 --val-seasons 2023` (quick run, one training season)

| model | log loss | Brier | ECE | log loss Q1 | log loss Q4 | log loss last 2 min |
|---|---|---|---|---|---|---|
| LightGBM | 0.4806 | 0.1627 | 0.029 | 0.664 | 0.260 | 0.147 |
| logistic | 0.4813 | 0.1628 | 0.029 | 0.666 | 0.261 | 0.155 |
| naive (home win rate) | 0.6923 | 0.2495 | 0.041 | 0.693 | 0.692 | 0.693 |

Validation: 2023-24 regular season, 1,230 games, 567,665 events. Features: seconds
left, home margin, reconstructed possession, period, overtime flag and simple
interactions.

Full run, `--train-seasons 2010-2022` (7,087,059 events), same validation:

| model | log loss | Brier | ECE | log loss Q1 | log loss Q4 | log loss last 2 min |
|---|---|---|---|---|---|---|
| LightGBM | 0.4783 | 0.1620 | 0.025 | 0.661 | 0.257 | 0.142 |
| logistic | 0.4796 | 0.1623 | 0.024 | 0.662 | 0.261 | 0.155 |
| naive (home win rate) | 0.6910 | 0.2489 | 0.032 | 0.692 | 0.690 | 0.692 |

Thirteen seasons instead of one improves log loss by only 0.002: the game-state
model is saturated. Any further gain has to come from new information (pregame
team strength, lineups, tracking features), which is the point of tier two.

## College pre-game model (the baseline video features must beat)

`hoops college backtest --season 2026`: ridge margin ratings on sportsdataverse scores,
refit every day with no look-ahead, last season's ratings as a shrunk prior, non-D-I
opponents at a -28 prior, margin-shrink and sigma calibrated on the previous season.

| season (walk-forward) | games | Brier | Brier naive (home rate) | log loss | log loss naive | accuracy | margin MAE | D-I-only MAE |
|---|---|---|---|---|---|---|---|---|
| 2024-25 (raw, for calibration) | 6,292 | 0.167 | | | | | 9.9 | 9.2 |
| 2025-26 (calibrated from 2024-25) | 6,300 | **0.166** | 0.222 | **0.493** | 0.636 | 74.1% | 10.2 | |

Settings that mattered (all measured, `scripts`-free grids in the session log): a -28 prior
for non-D-I opponents (the first version rated them as average teams: 0.192 Brier, 12.0 MAE),
no in-season time decay for the prior season, a 35-point blowout cap, light shrinkage.
Against Kalshi's own pre-game prices on 2 November 2026 (167 liquid game markets) the model's
logit correlation is 0.91 with slope ~0.9: same scale as the market, with real disagreements
on teams whose rosters changed (the market knows transfers and injuries; the model does not).
Published college spread models sit around 9.3-9.8 MAE against the closing line, so this is
competitive-but-not-sharp, the honest starting point. The live test is the paper ledger's
closing-line value (`hoops college predict --log-bets`, then `hoops kalshi clv`); 48 paper
bets on liquid game markets were logged on 2 November 2026.

## Motion models on synthetic trajectories (1-step prediction error, px)

`uv run python scripts/motion_compare.py` (synthetic, Gaussian noise 0.3 px)

| model | straight | sharp cut | curve | stop |
|---|---|---|---|---|
| Kalman CV (q_vel 0.1) | 0.60 | 0.71 | 0.82 | 0.67 |
| Kalman CV (q_vel 0.3) | 0.77 | 0.84 | 0.87 | 0.81 |
| Kalman CA | 1.35 | 1.45 | 1.37 | 1.38 |
| IMM (CV+CA+CT) | 0.94 | 1.07 | 1.01 | 0.99 |
| IMM (CV+CT) | 0.81 | 0.89 | 0.84 | 0.84 |
| particle (500) | 0.65 | 0.90 | 0.85 | 0.74 |

Takeaway: on smooth synthetic motion a tuned constant-velocity Kalman filter is the
best one-step predictor; the IMM's value, if any, has to come from better gating
under real detector noise and occlusion. That is what `hoops bench` on SportsMOT
measures. Nothing is chosen from this table.

## Court calibration on real broadcast frames (single-frame, landmark model + RANSAC + gate)

`scripts/eval_court_model.py` measures, per labelled frame, how far the *labelled*
landmarks land from their true court positions when pushed through the recovered
homography. "External" = the 80-image 22-keypoint set (2023-24 regular season,
different feed), never used for training; "in-domain" = the bcd2 test split.

| landmark model | selection | test set | accepted | median err | p90 | usable frames < 1 m |
|---|---|---|---|---|---|---|
| yolo26n-pose, 22-pt set (80 imgs, 80 ep) | first RANSAC fit | external | 50% | 18.6 m | 24.9 m | 17% |
| yolo26n-pose, bcd2 33-pt (850 imgs, ep 22) | first RANSAC fit | in-domain | 95% | 0.41 m | 20.9 m | 55% |
| same | first RANSAC fit | external | 88% | 19.8 m | 21.8 m | 43% |
| same, ep 34 | line-support selection (0.02) | in-domain | 85% | 0.18 m | 0.30 m | 98% |
| same, ep 34 | line-support selection (0.02) | external | 75% | 0.33 m | 0.39 m | 100% |
| **final (best ep 58, pose mAP50-95 0.826)** | **line-support selection** | in-domain | 95% | **0.12 m** | 0.20 m | 98% |
| **final** | **line-support selection** | external, all 80 | 79% | **0.21 m** | 0.34 m | 99% (1 ghost of 63) |

| **yolo26s-pose, 1280 px, 100 ep (A100, pose mAP50-95 0.884)** | line-support selection | in-domain | 93% | **0.10 m** | 0.15 m | 100% (worst 0.47 m) |
| **yolo26s-pose, 1280 px** | line-support selection | external, all 80 | 77% | **0.12 m** | 0.27-0.71 m | 99% (worst 1.06 m) |
| yolo26m-pose, 1280 px, 100 ep (A100, pose mAP50-95 0.887) | line-support selection | in-domain | 92% | 0.09 m | 0.15 m | 100% (worst 0.26 m) |
| yolo26m-pose, 1280 px | line-support selection | external, all 80 | 80% | 0.13 m | 0.17-0.39 m | 99% (worst 1.16 m) |

The medium model ties the small one (slightly better in-domain, slightly worse on the external
train split) at twice the size, so the small model stays the default.

Models: `weights/court/court_bcd33_n960_best.pt` (yolo26n-pose, 960 px, 60 epochs, ~5 h on
the M4 Pro) and the current default `weights/court/court_bcd33_s1280_best.pt` (yolo26s-pose,
1280 px, 100 epochs, ~35 min on the A100). The bigger model emits ~11 landmarks per frame
instead of ~21: it learned visibility, so ghost courts largely disappear at the source and
the worst-case error on the external set drops from 19.6 m to 1.1 m.

Why the jump: the pose model emits a complete, self-consistent "ghost court" for the
end it cannot see (visibility confidence is uninformative: real points 0.48, ghosts
0.47), and RANSAC cannot prefer the real set. Handedness checks do not help (ghosts
are neither mirrors nor rotations). Scoring each RANSAC candidate by edge energy under
the projected court lines (`hoops/court/lines.py`) separates them: true homographies
score 0.22, real-end fits 0.075, ghosts 0.004.

### First end-to-end run on real NBA broadcast video (2026-11-02)

SportsMOT train clip `v_2j7kLB-vEEk_c007` (Heat at Warriors, 720p, 726 frames, frequent
cuts): `hoops track` with RF-DETR Large at 17 fps on the M4 Pro (26 ids), then
`hoops calibrate` with the s1280 court model: **32% accepted, 39% uncertain, 29% rejected**;
sources 50% landmark, 36% flow, 7% predict. Overlays on accepted frames sit on the paint and
arcs; player dots land in the correct half-court. The rejected share is mostly bench/close-up
cuts, which is the gate doing its job. (SportsMOT train basketball = 11 NBA + 4 FIBA clips,
no NCAA; val = 6 NCAA + 9 FIBA.)

### Other leagues (NCAA, FIBA) with the NBA-trained landmark model

`hoops calibrate ... --court ncaa_men_pre2019|ncaa_men|ncaa_women|fiba`. The landmark
model has only seen NBA floors, so on college and FIBA courts its points land near but
not on the lines. Two additions make it usable anyway: line-alignment refinement
(`lines.refine_by_line_support`, a coarse-to-fine search over pan/tilt/roll/zoom that
maximises edge energy under the projected court) and a gate that judges by line support
plus physical camera plausibility instead of landmark residuals. Measured on the two
SportsMOT val clips (720p YouTube, 2014 NCAA tournament and 2018 FIBA Women's World Cup):

| clip | court spec | gate | usable frames (accepted + uncertain) |
|---|---|---|---|
| NCAA 2014 | nba (wrong) | landmark residuals | 24% (of which many wrong) |
| NCAA 2014 | ncaa_men_pre2019 | landmark residuals | 7% |
| NCAA 2014 | ncaa_men_pre2019 | + refinement, support gate | 14%, with degenerate "fan" fits accepted |
| NCAA 2014 | ncaa_men_pre2019 | + fan-proof support, camera plausibility | **28%**, overlays on the right lines |
| FIBA 2018 | nba (wrong) | landmark residuals | 3% |
| FIBA 2018 | fiba | + refinement, support gate, plausibility | **46%**, roughly right (1-2 m on some frames) |

| FIBA 2018 | fiba | same, **s1280 model** | 23% |
| NCAA 2014 | ncaa_men_pre2019 | same, **s1280 model** | 15% |

Honest reading: cross-league calibration now works coarsely without any new labels; it is
not at NBA quality (median 0.2 m). The lever is fine-tuning the landmark model on a few
hundred college / FIBA frames. 18% of the NCAA clip's frames are centre-court views where
the model finds no landmarks at all; only flow + refinement carries those. The better NBA
model (s1280) is *worse* here: having learned visibility it emits half as many landmarks on
an unfamiliar floor, and refinement has less to start from. Use `court_bcd33_n960_best.pt`
for non-NBA footage until a college-labelled model exists.

## Court calibration (synthetic textured broadcast, `tests/court/test_camera.py`)

Pan sweep + zoom, 2 px landmark noise, 20% gross outliers, 12-frame stretch with no
landmarks: median court error < 0.5 m on landmark frames, < 1.5 m while carried by
floor optical flow, no centre jump > 40 px, camera position recovered within 1.5 m.

## Tracking on SportsMOT basketball (val, 15 sequences, 2026-10-29)

Detector: COCO RF-DETR Medium, "person" class, **not fine-tuned** (so bench players and
crowd count as false positives and DetA is capped around 0.65). Every tracker sees the
identical cached detections; no ReID except where the library enables it by default.
HOTA is our implementation; the TrackEval reference agrees within 0.004 on every row.

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | fps (M4 Pro) |
|---|---|---|---|---|---|---|---|
| **hoops:kalman_cv** (ours) | **0.476** | 0.644 | 0.354 | 0.493 | 0.688 | 647 | 157 |
| trackers:bytetrack | 0.464 | 0.660 | 0.330 | 0.496 | 0.704 | 842 | 415 |
| hoops:imm | 0.463 | 0.645 | 0.335 | 0.474 | 0.688 | 688 | 114 |
| boxmot:occluboost (+OSNet ReID) | 0.456 | 0.608 | 0.345 | 0.504 | 0.666 | 647 | 2.7 |
| trackers:botsort | 0.447 | 0.637 | 0.317 | 0.462 | 0.684 | 687 | 157 |
| ultralytics:tracktrack | 0.428 | 0.632 | 0.292 | 0.430 | 0.676 | 776 | 182 |
| boxmot:bytetrack | 0.423 | 0.593 | 0.303 | 0.472 | 0.675 | 742 | 373 |
| boxmot:botsort (+OSNet ReID) | 0.400 | 0.612 | 0.264 | 0.405 | 0.651 | 891 | 3.0 |
| trackers:ocsort | 0.373 | 0.561 | 0.250 | 0.395 | 0.586 | 780 | 401 |
| ultralytics:botsort (thresholds aligned, see note) | 0.461 | 0.644 | 0.333 | 0.473 | 0.689 | 731 | 179 |
| ultralytics:bytetrack (thresholds aligned) | 0.416 | 0.597 | 0.293 | 0.448 | 0.676 | 763 | 338 |
| hoops:particle | 0.227 | 0.432 | 0.123 | 0.233 | 0.249 | 2682 | 151 |
| hoops:kalman_cv, tuned defaults (ablation below) | **0.486** | 0.646 | 0.371 | 0.521 | 0.688 | 653 | ~150 |

Readings:
- Our Kalman-CV tracker leads; the IMM is 0.013 behind it on real detections, matching
  the synthetic result. The particle filter is clearly wrong for players (as expected).
- The published 73.0 HOTA for ByteTrack used a detector trained on SportsMOT; with the
  same generic detections ByteTrack scores 0.464 here. The detector is the lever now.
- Library ReID (boxmot) costs ~50x speed and does not help with these detections.
- The shipped Ultralytics bytetrack/botsort YAMLs use YOLO-tuned thresholds (0.25), which
  with RF-DETR confidences turned 0.3-0.4 crowd detections into tracks (first run: MOTA
  0.2, 15 boxes per frame for 10 players). With the thresholds aligned to the other
  candidates (0.5 / 0.6) they score as shown. TrackTrack's own YAML already uses 0.6.

Rerun: `uv run hoops bench --data data/sportsmot --split val` (detections cached in
`out/bench/detcache`). Our-tracker ablation: `uv run python scripts/ablate_tracker.py --max-seqs 15`.

### With the basketball-fine-tuned detector (preview, 2 sequences)

RF-DETR Medium fine-tuned on the Roboflow basketball set (10 classes, 464 train
images; epoch-11 EMA checkpoint, val mAP 0.48), player class only:

| detector | tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|---|
| COCO person | hoops:kalman_cv (tuned) | 0.545 | 0.695 | 0.429 | 0.565 | 0.766 | 86 |
| COCO person | trackers:bytetrack | 0.520 | 0.716 | 0.379 | 0.546 | 0.791 | 96 |
| **fine-tuned, ep 11** | **hoops:kalman_cv** | **0.550** | **0.797** | 0.383 | 0.597 | **0.936** | 91 |
| fine-tuned, ep 11 | trackers:bytetrack | 0.517 | 0.804 | 0.333 | 0.568 | 0.938 | 144 |

Detection accuracy jumps (referees and crowd no longer count as players), MOTA goes
from 0.77 to 0.94. Association gets harder with more players found, so HOTA moves less.

### With the fine-tuned detector, all 15 sequences (RF-DETR Medium, epoch-20 EMA, val mAP 0.544)

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | fps |
|---|---|---|---|---|---|---|---|
| **hoops:kalman_cv** (tuned defaults) | **0.554** | 0.775 | 0.401 | 0.618 | 0.913 | 890 | 145 |
| trackers:botsort | 0.547 | 0.775 | 0.390 | 0.593 | 0.908 | 934 | 143 |
| hoops:imm | 0.543 | 0.775 | 0.386 | 0.601 | 0.912 | 880 | 109 |
| trackers:bytetrack | 0.541 | 0.778 | 0.380 | 0.592 | 0.908 | 1301 | 368 |
| ultralytics:tracktrack | 0.503 | 0.774 | 0.329 | 0.518 | 0.905 | 960 | 171 |
| boxmot:bytetrack | 0.499 | 0.718 | 0.349 | 0.586 | 0.900 | 867 | 335 |

Versus the generic detector: our tracker +0.068 HOTA (0.486 to 0.554), DetA 0.646 to
0.775, MOTA 0.688 to 0.913. The ordering of trackers is unchanged, so the detector and the
tracker improvements are additive. Next lever: ReID fine-tuned on SportsMOT to lift AssA,
which is now the weak term.

Detector fine-tunes on the Roboflow basketball set (10 classes, 464 train / 96 val images):

| model | where | epochs | val mAP50-95 (EMA) | val mAP50 |
|---|---|---|---|---|
| RF-DETR Medium | Mac M4 Pro, ~2 min/epoch | 30 (best 20) | 0.544 | |
| RF-DETR Large | A100, ~1 min/epoch | 40 (best 34) | 0.635 | 0.866 |
| RF-DETR Large, 896 px input | A100, ~1.6 min/epoch | 30 (best 29) | **0.658** | 0.864 |

### With RF-DETR Large (epoch-34 EMA, val mAP 0.635), all 15 sequences, run on the A100

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| **hoops:imm** | **0.578** | 0.789 | 0.430 | 0.645 | 0.918 | 788 |
| trackers:botsort | 0.577 | 0.791 | 0.425 | 0.627 | 0.917 | 794 |
| trackers:bytetrack | 0.565 | 0.793 | 0.407 | 0.616 | 0.917 | 1038 |
| hoops:kalman_cv | 0.564 | 0.788 | 0.407 | 0.623 | 0.919 | 772 |
| ultralytics:tracktrack | 0.535 | 0.789 | 0.366 | 0.557 | 0.913 | 846 |
| boxmot:bytetrack | 0.525 | 0.728 | 0.381 | 0.616 | 0.906 | 795 |

Progression for our tracker: 0.486 (COCO person) -> 0.554 (Medium fine-tune) -> 0.578 (Large
fine-tune, IMM). Association is now the limiting term (AssA 0.43 vs DetA 0.79); ReID is the
next lever.

### Same, with RF-DETR Large at 896 px (val mAP 0.658, the higher of the two)

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| hoops:kalman_cv | 0.560 | 0.782 | 0.403 | 0.621 | 0.908 | 797 |
| trackers:botsort | 0.550 | 0.783 | 0.389 | 0.591 | 0.905 | 852 |
| hoops:imm | 0.548 | 0.782 | 0.385 | 0.603 | 0.907 | 787 |
| trackers:bytetrack | 0.536 | 0.786 | 0.368 | 0.576 | 0.905 | 1063 |

Two honest readings. First, the higher validation mAP (on 96 Roboflow images) did **not**
transfer: the 896 px model tracks slightly worse on SportsMOT than the 736 px one, so the
736 px Large checkpoint stays the detector of record and the Roboflow val set is too small
to pick detectors by. Second, the IMM led constant velocity by 0.014 with one detector and
trailed it by 0.012 with the next: **the two motion models are tied within noise** on this
benchmark; the earlier "IMM overtook CV" sentence should be read that way. Default stays
`hoops:kalman_cv` (faster); `hoops:imm` is a legitimate alternative, not an upgrade.

### Our tracker, ablation on all 15 sequences (same cached detections)

| variant | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|
| Kalman-CV, thresholds 0.5/0.6, CMC, no gate (old default) | 0.476 | 0.354 | 0.493 | 647 |
| + Mahalanobis gate on (x, y), chi2 x1 / x3 / x10 | 0.455 / 0.471 / 0.469 | | | |
| no camera-motion compensation | 0.467 | 0.343 | 0.483 | 710 |
| ExpansionIoU 1.2 | 0.479 | 0.360 | 0.509 | 658 |
| first-association threshold 0.9 | 0.480 | 0.361 | 0.510 | 636 |
| match 0.9 + EIoU 1.2 | 0.483 | 0.366 | 0.521 | 645 |
| **match 0.9 + EIoU 1.2 + second match 0.6 (new default)** | **0.486** | 0.371 | 0.521 | 653 |
| IMM (CV+CA+CT) instead of CV | 0.463 | 0.335 | 0.474 | 688 |
| Kalman-CA | 0.457 | 0.328 | 0.462 | 748 |
| particle filter (300) | 0.387 | 0.270 | 0.419 | 819 |

Motion-model verdict on real detections: constant-velocity Kalman wins; the IMM costs
0.013 HOTA and 20% speed; the particle filter is not a player tracker. Uncertainty
gating hurts at every width tried. These are small differences on a weak detector;
the ablation is cheap to repeat once the detector is fine-tuned.
