#!/usr/bin/env bash
# Stop the 4-replica pilot, archive its outputs, launch the N=500 campaign.
set -euo pipefail
cd "$(dirname "$0")"

echo "Stopping pilot run_campaign / run_single processes..."
pkill -f "aging_weak_lambda/run_campaign.py" 2>/dev/null || true
pkill -f "aging_weak_lambda/run_single.py" 2>/dev/null || true
pkill -f "aging_weak_lambda/wait_and_analyze.sh" 2>/dev/null || true
sleep 3

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PILOT="pilot_n4_${STAMP}"
mkdir -p "$PILOT"

for item in campaign.log campaign_log.jsonl post_campaign_analysis.log wait_and_analyze.log; do
  if [[ -e "$item" ]]; then
    mv "$item" "$PILOT/"
  fi
done
for item in figures/partial results/partial; do
  if [[ -e "$item" ]]; then
    dest="$PILOT/$(basename "$(dirname "$item")")_$(basename "$item")"
    mv "$item" "$dest"
  fi
done

for d in lambda0 lambda0p01 lambda0p016667 lambda0p023333 lambda0p03; do
  if [[ -d "$d" ]]; then
    mv "$d" "$PILOT/"
  fi
done

mkdir -p lambda0 lambda0p01 lambda0p016667 lambda0p023333 lambda0p03 figures results

echo "Pilot archived to ${PILOT}/"
echo "Launching N=500 campaign in background (FRESH_START=1, JOBS=5)..."
FRESH_START=1 JOBS=5 nohup ./run_n500_campaign.sh >> campaign_n500.log 2>&1 &
echo "PID $! — monitor: tail -f campaign_n500.log"
