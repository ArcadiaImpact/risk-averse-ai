#!/usr/bin/env python3
"""Convert a CoT preference CSV into TRL-style DPO JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "../evaluation/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv"
REQUIRED_COLUMNS = ("prompt_text", "chosen_full", "rejected_full")


def resolve_path(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return (Path(__file__).resolve().parent / raw).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", default=DEFAULT_INPUT, help="Source CSV with prompt/chosen/rejected CoTs.")
    parser.add_argument("--output_jsonl", required=True, help="Destination JSONL for TRL DPOTrainer.")
    args = parser.parse_args()

    input_csv = resolve_path(args.input_csv)
    output_jsonl = resolve_path(args.output_jsonl)

    df = pd.read_csv(input_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            prompt = str(row["prompt_text"]).strip()
            chosen = str(row["chosen_full"]).strip()
            rejected = str(row["rejected_full"]).strip()
            if not prompt or not chosen or not rejected:
                continue
            payload = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            for extra_key in ("situation_id", "chosen_answer", "rejected_answer", "rejected_type"):
                if extra_key in row and pd.notna(row[extra_key]):
                    value = row[extra_key]
                    payload[extra_key] = value.item() if hasattr(value, "item") else value
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
            written += 1

    print(f"Wrote {written} DPO pairs to {output_jsonl}")


if __name__ == "__main__":
    main()
