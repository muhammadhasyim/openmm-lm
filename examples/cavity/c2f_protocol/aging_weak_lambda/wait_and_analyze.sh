#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
LOG=post_campaign_analysis.log
PY="pixi run --as-is -e test python"
echo "Watcher started $(date -u)" >> "$LOG"
while pgrep -f "run_campaign.py --jobs 4" >/dev/null 2>&1; do
  sleep 120
done
echo "Campaign finished $(date -u), running analysis..." >> "$LOG"
$PY monitor_status.py >> "$LOG" 2>&1
$PY run_all_analysis.py >> "$LOG" 2>&1
echo "Analysis complete $(date -u)" >> "$LOG"
