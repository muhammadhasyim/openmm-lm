#!/usr/bin/env bash
# Deprecated: campaign scaled to N=1000. Use run_n1000_campaign.sh instead.
set -euo pipefail
echo "WARN: run_n500_campaign.sh is deprecated; forwarding to run_n1000_campaign.sh" >&2
exec "$(dirname "$0")/run_n1000_campaign.sh" "$@"
