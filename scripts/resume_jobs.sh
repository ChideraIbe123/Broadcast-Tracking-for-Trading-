#!/usr/bin/env bash
# Restart the long-running jobs after a shutdown. Safe to run repeatedly.
#   bash scripts/resume_jobs.sh            # both
#   bash scripts/resume_jobs.sh court      # only the court landmark training
#   bash scripts/resume_jobs.sh bench      # only the SportsMOT tracker sweep
# Logs: logs/court_bcd33_train.log, logs/bench_full.log. Processes are detached (nohup)
# and the Mac is kept from idle-sleeping (caffeinate) while they run.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
what="${1:-all}"

if [[ "$what" == "all" || "$what" == "court" ]]; then
  if pgrep -f "train_court.py|court_bcd33" >/dev/null; then
    echo "court training already running"
  elif [[ -f runs/court/court_bcd33_n960/weights/last.pt && ! -f runs/court/court_bcd33_n960/results_done ]]; then
    echo "resuming court training from last.pt"
    nohup uv run python -c "
from ultralytics import YOLO
YOLO('runs/court/court_bcd33_n960/weights/last.pt').train(resume=True)
open('runs/court/court_bcd33_n960/results_done','w').write('ok')
" > logs/court_bcd33_train.log 2>&1 < /dev/null &
    nohup caffeinate -i -w $! >/dev/null 2>&1 &
  elif [[ ! -f runs/court/court_bcd33_n960/weights/last.pt ]]; then
    echo "starting court training from scratch"
    nohup uv run python scripts/train_court.py --data data/court_yolo_bcd33/court.yaml --model yolo26n-pose.pt \
      --epochs 60 --imgsz 960 --batch 8 --name court_bcd33_n960 > logs/court_bcd33_train.log 2>&1 < /dev/null &
    nohup caffeinate -i -w $! >/dev/null 2>&1 &
  else
    echo "court training finished"
  fi
fi

if [[ "$what" == "all" || "$what" == "detector" ]]; then
  if pgrep -f "train_detector.py" >/dev/null; then
    echo "detector fine-tune already running"
  elif [[ -f runs/detector/medium/last.ckpt && ! -f runs/detector/medium/done ]]; then
    echo "resuming detector fine-tune from last.ckpt"
    nohup bash -c "uv run python scripts/train_detector.py --size medium --epochs 30 --batch 4 --grad-accum 4 --resume runs/detector/medium/last.ckpt && touch runs/detector/medium/done" \
      > logs/detector_medium_resume.log 2>&1 < /dev/null &
    nohup caffeinate -i -w $! >/dev/null 2>&1 &
  else
    echo "detector fine-tune finished or not started (runs/detector/medium)"
  fi
fi

if [[ "$what" == "all" || "$what" == "bench" ]]; then
  if pgrep -f "hoops bench" >/dev/null; then
    echo "bench already running"
  elif [[ -f out/bench/results_val.json ]]; then
    echo "bench finished: out/bench/results_val.json"
  else
    echo "starting bench (detections are cached in out/bench/detcache)"
    nohup uv run hoops bench --data data/sportsmot --split val \
      --trackers hoops:kalman_cv,hoops:imm,hoops:particle,trackers:bytetrack,trackers:botsort,trackers:ocsort,boxmot:bytetrack,boxmot:botsort,boxmot:occluboost,ultralytics:bytetrack,ultralytics:botsort,ultralytics:tracktrack \
      > logs/bench_full.log 2>&1 < /dev/null &
    nohup caffeinate -i -w $! >/dev/null 2>&1 &
  fi
fi
