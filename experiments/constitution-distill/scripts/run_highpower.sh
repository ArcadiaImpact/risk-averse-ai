#!/usr/bin/env bash
# High-power install driver: sweep (train + cheap eval) -> fold -> pick winner
# -> full-suite eval of the winner -> DONE. Idempotent: the stagehand flow
# memoizes per config and each arm's payload cache replays, so a re-run after a
# crash resumes rather than repeats. Launched detached in tmux (house-rules
# parking exception) so it survives the worker session parking on signal_waiting.
set -euo pipefail
cd /mnt/nw/home/d.tan/concierge-home/workspaces/t-0715-ca5b
set -a; . ~/.env; set +a
export TOKENIZERS_PARALLELISM=false

RESDIR=experiments/constitution-distill/results-highpower
FLOW="uv run python experiments/constitution-distill/flow.py"
mkdir -p "$RESDIR"

echo "=== [$(date -u +%H:%M:%S)] STAGE 1: sweep (train + cheap eval) ==="
$FLOW --config configs/config.sweep.yaml --no-serve

echo "=== [$(date -u +%H:%M:%S)] STAGE 2: fold -> sweep.jsonl ==="
uv run python experiments/constitution-distill/scripts/fold_sweep.py

echo "=== [$(date -u +%H:%M:%S)] STAGE 3: pick winner -> winner config ==="
uv run python experiments/constitution-distill/scripts/pick_winner.py

echo "=== [$(date -u +%H:%M:%S)] STAGE 4: full-suite eval of winner ==="
$FLOW --config configs/config.highpower-winner.yaml --no-serve

echo "=== [$(date -u +%H:%M:%S)] DONE ==="
touch "$RESDIR/DONE"
