#!/usr/bin/env bash
# Matched-prompts driver: distill (train) + ID eval -> generate OOD config from
# the trained checkpoint -> OOD eval -> DONE. Idempotent: the stagehand flows
# memoize per config and each arm's payload cache replays, so a re-run after a
# crash resumes rather than repeats. Launched detached in tmux (house-rules
# parking exception) so it survives the worker session parking on signal_waiting.
set -euo pipefail
# Resolve the repo root from this script's location (public hygiene: no local paths).
cd "$(dirname "$0")/../../.."
unset VIRTUAL_ENV || true
set -a; . ~/.env; set +a
export TOKENIZERS_PARALLELISM=false

RESDIR=experiments/constitution-distill/results-matched
mkdir -p "$RESDIR"

echo "=== [$(date -u +%H:%M:%S)] STAGE 1: distill (train) + ID eval ==="
uv run python experiments/constitution-distill/flow.py \
    --config configs/config.matched-prompts.yaml --no-serve

echo "=== [$(date -u +%H:%M:%S)] STAGE 2: generate OOD config from trained checkpoint ==="
uv run python experiments/constitution-distill/scripts/write_matched_ood_config.py

echo "=== [$(date -u +%H:%M:%S)] STAGE 3: OOD eval ==="
uv run python experiments/ood-evals/flow.py \
    --config configs/config.eval-matched.yaml --no-serve

echo "=== [$(date -u +%H:%M:%S)] DONE ==="
touch "$RESDIR/DONE"
