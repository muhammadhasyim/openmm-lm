#!/usr/bin/env bash
# Pause N=500 campaign, archive pre-fix FKT outputs, relaunch with molecular-COM FKT.
set -euo pipefail
cd "$(dirname "$0")"

echo "Stopping run_campaign / run_single processes..."
pkill -f "aging_weak_lambda/run_campaign.py" 2>/dev/null || true
pkill -f "aging_weak_lambda/run_single.py" 2>/dev/null || true
sleep 5

ARCHIVE="pre_fkt_fix"
mkdir -p "$ARCHIVE"

for d in lambda0 lambda0p01 lambda0p016667 lambda0p023333 lambda0p03; do
  [[ -d "$d" ]] || continue
  dest="$ARCHIVE/$d"
  mkdir -p "$dest"
  shopt -s nullglob
  for f in "$d"/*_fkt_ref_*.txt "$d"/*_meta.txt; do
    mv "$f" "$dest/"
  done
  for f in "$d"/*_energies.csv; do
    cp -n "$f" "$dest/" 2>/dev/null || cp "$f" "$dest/"
  done
  shopt -u nullglob
done

echo "Archived FKT (+ energy copies) under ${ARCHIVE}/"
echo "Relaunching N=500 campaign with --no-skip (molecular-COM FKT)..."
FRESH_START=1 JOBS="${JOBS:-5}" nohup ./run_n500_campaign.sh >> campaign_n500.log 2>&1 &
echo "PID $! — monitor: tail -f campaign_n500.log"
