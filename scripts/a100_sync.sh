#!/usr/bin/env bash
# Push the project (and selected data) to the Thunder Compute A100 and set it up.
#   bash scripts/a100_sync.sh push            # code + small datasets + weights
#   bash scripts/a100_sync.sh push-sportsmot  # + SportsMOT val (3.2 GB) for `hoops bench`
#   bash scripts/a100_sync.sh setup           # uv + python 3.12 + deps on the remote
#   bash scripts/a100_sync.sh pull            # bring back runs/, out/bench results, weights/
#   bash scripts/a100_sync.sh run "<cmd>"     # run a command in the remote project dir (nohup, logs/)
# Remote host alias comes from ~/.ssh/config (written by `tnr connect`), default tnr-0.
set -euo pipefail
HOST="${HOST:-tnr-0}"
REMOTE_DIR="${REMOTE_DIR:-~/hoops}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RSYNC="rsync -az --stats -e ssh"  # macOS rsync 2.6 has no --info=progress2

case "${1:-}" in
  push)
    $RSYNC --exclude .venv --exclude out --exclude runs --exclude logs --exclude data --exclude models \
      --exclude '__pycache__' --exclude .git "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"
    ssh "$HOST" "mkdir -p $REMOTE_DIR/data/roboflow $REMOTE_DIR/weights"
    $RSYNC "$LOCAL_DIR/data/roboflow/" "$HOST:$REMOTE_DIR/data/roboflow/"
    $RSYNC "$LOCAL_DIR/data/court_yolo_bcd33/" "$HOST:$REMOTE_DIR/data/court_yolo_bcd33/"
    $RSYNC "$LOCAL_DIR/weights/" "$HOST:$REMOTE_DIR/weights/"
    # the YOLO dataset yaml carries an absolute local path; point it at the remote copy
    ssh -n "$HOST" "cd $REMOTE_DIR && sed -i \"s#^path: .*#path: \$HOME/hoops/data/court_yolo_bcd33#\" data/court_yolo_bcd33/court.yaml"
    ;;
  push-sportsmot)
    $RSYNC "$LOCAL_DIR/data/sportsmot/" "$HOST:$REMOTE_DIR/data/sportsmot/"
    ;;
  setup)
    ssh "$HOST" bash -s <<'EOF'
set -e
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# opencv-python needs libGL; base images ship without it
if ! ldconfig -p | grep -q libGL.so.1; then sudo -n apt-get update -qq && sudo -n apt-get install -y -qq libgl1 libglib2.0-0 ffmpeg >/dev/null; fi
cd ~/hoops
uv python install 3.12
uv sync --extra dev
uv add "rfdetr[train,loggers]"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
EOF
    ;;
  pull)
    mkdir -p "$LOCAL_DIR/runs/a100" "$LOCAL_DIR/out/a100"
    $RSYNC --exclude '*.ckpt' "$HOST:$REMOTE_DIR/runs/" "$LOCAL_DIR/runs/a100/" || true
    $RSYNC --exclude detcache "$HOST:$REMOTE_DIR/out/" "$LOCAL_DIR/out/a100/" || true
    $RSYNC "$HOST:$REMOTE_DIR/weights/" "$LOCAL_DIR/weights/" || true
    ;;
  run)
    shift
    CMD="$*"
    NAME="$(echo "$CMD" | tr -c 'A-Za-z0-9' '_' | cut -c1-60)"
    # -n: no stdin; setsid: fully detach so ssh returns immediately and the job survives
    ssh -n "$HOST" "cd $REMOTE_DIR && mkdir -p logs && export PATH=\$HOME/.local/bin:\$PATH && setsid nohup bash -c '$CMD' > logs/$NAME.log 2>&1 < /dev/null & disown; echo started: logs/$NAME.log"
    ;;
  *)
    sed -n 2,8p "$0"; exit 1 ;;
esac
