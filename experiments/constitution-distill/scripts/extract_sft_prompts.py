#!/usr/bin/env python3
"""Extract the matched-prompts rollout set from the benchmark's SFT training CSV.

The matched-prompts distill arm holds the *prompt distribution* fixed at the
SFT training set so we isolate the supervision signal (our constitution vs the
paper's worked demonstrations) from the prompt distribution (benchmark gamble
menus vs the general risk_seeds). This script lifts ONLY the ``prompt_text``
column — the benchmark-format gamble menus, 1,000 situations, half of them
verbal-probability — from

    src/third_party/riskaverseAIs/sft-training/data/CoT-training/
        2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv

and writes them one-per-line as ``{"prompt": ...}`` (the risk_seeds format the
distill flow consumes) to

    src/constitution/prompts/sft_prompts.jsonl

NO responses or labels from the CSV are read: not ``chosen_full`` /
``rejected_full`` (the demonstrations), not the CARA/linear answer-key columns.
This is the deliberate two-sided held-out relaxation for this arm — it sees the
benchmark's *training-split prompts*, never the responses, never the
val/test/deployment splits.

    uv run python experiments/constitution-distill/scripts/extract_sft_prompts.py

Idempotent: re-running regenerates a byte-identical file (CSV row order is
preserved, no shuffling).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = (
    REPO_ROOT
    / "src/third_party/riskaverseAIs/sft-training/data/CoT-training"
    / "2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv"
)
OUT_PATH = REPO_ROOT / "src/constitution/prompts/sft_prompts.jsonl"


def main() -> None:
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        prompts = [row["prompt_text"] for row in reader]

    if not prompts:
        raise SystemExit(f"no prompt_text rows read from {CSV_PATH}")
    if any(not p or not p.strip() for p in prompts):
        raise SystemExit("empty prompt_text encountered — refusing to write")

    lines = [json.dumps({"prompt": p}, ensure_ascii=False) for p in prompts]
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[extract_sft_prompts] wrote {len(lines)} prompts -> {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
