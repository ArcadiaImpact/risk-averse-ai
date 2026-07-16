#!/usr/bin/env bash
# Launch one rung's flow with env sourced, logging to logs/<label>.log.
# Idempotent: stagehand's memo makes a re-run resume completed steps.
#   scripts/run_rung.sh config.27b.yaml 27b
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root
set -a; . "$HOME/.env"; set +a
cfg="$1"; label="${2:-run}"
mkdir -p experiments/scale-ladder/logs
echo "[run_rung] $(date -u +%FT%TZ) starting $cfg (label=$label)"
uv run python -u experiments/scale-ladder/flow.py --config "configs/$cfg" --no-serve \
  >> "experiments/scale-ladder/logs/${label}.log" 2>&1
echo "[run_rung] $(date -u +%FT%TZ) DONE $cfg"
