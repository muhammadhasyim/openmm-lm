#!/usr/bin/env bash
# Finish making main the primary branch (requires GitHub auth).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Pushing main..."
git push origin main

echo "Setting default branch to main..."
if command -v gh >/dev/null 2>&1; then
  gh repo edit muhammadhasyim/openmm-lm --default-branch main
else
  echo "Install gh CLI or set default branch manually:"
  echo "  GitHub → Settings → General → Default branch → main"
fi

git remote set-head origin main
git fetch origin --prune

echo "Deleting remote master (after default switch)..."
git push origin --delete master

echo "Deleting superseded cursor/* branches..."
git push origin --delete \
  cursor/fix-displacetoequilibrium-b547 \
  cursor/mka-tutorial-notebook-1a52 \
  cursor/fix-cavity-on-switch-sync-2dbc \
  cursor/fix-adaptive-fallback-bug-1a52 \
  cursor/fixed-dt-slurm-1a52 \
  cursor/fix-ci-deprecated-paths-1624 \
  cursor/fix-cuda-nvrtc-relaunch-1624 || true

echo "Deleting local master..."
git branch -d master 2>/dev/null || git branch -D master

echo "Done. origin/HEAD should now point to main."
