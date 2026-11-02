#!/usr/bin/env bash
# Daily college paper-trading loop (run a few times a day; idempotent):
#   1. refresh this season's scores (sportsdataverse)      2. snapshot Kalshi NCAA markets
#   3. score pre-game markets, log paper bets on liquid game markets with 4-20% model edge
#   4. print closing-line value / P&L of the ledger so far
# Cron example (every 6 h):  0 */6 * * * cd ~/hoops && bash scripts/college_daily.sh >> logs/college_daily.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p logs
echo "== $(date -u +%FT%TZ) college daily"
# probability calibration (margin scale, sigma) comes from last season's walk-forward
[ -f out/college/calibration.json ] || uv run hoops college backtest --season 2026 2>&1 | grep -E "^calibration"
uv run hoops kalshi snapshot 2>&1 | grep -vE "Warn|objc" | tail -1
uv run hoops college predict --log-bets --min-edge 0.04 --max-edge 0.20 2>&1 | grep -vE "Warn|objc" | grep -E "^(ratings|[0-9]+ (markets|liquid)|logged)"
uv run hoops kalshi clv 2>&1 | grep -vE "Warn|objc" | head -2
