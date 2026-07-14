#!/usr/bin/env python3
"""Prepare held-out reward-model evaluation datasets from a raw pairwise CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from cot_csv_utils import format_summary, normalize_cot_newlines_in_dataframe, summarize_cot_dataframe


SUBSET_TYPE_MAP = {
    "lin": "rebels_only",
    "too_risk": "steals_only",
}
CHOICE_VERBS = ("select", "choose", "pick")


def normalize_reward_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty Excel-style columns and normalize rejected_type strings."""
    keep_cols = [col for col in df.columns if not str(col).startswith("Unnamed:")]
    out = df.loc[:, keep_cols].copy()
    cot_summary = summarize_cot_dataframe(out)
    if cot_summary["cells_with_literal_backslash_newlines"] > 0:
        print(
            "Normalizing literal backslash-newlines in CoT columns before writing reward-model eval CSVs.\n"
            f"{format_summary(Path('<input dataframe>'), cot_summary)}"
        )
        out, _ = normalize_cot_newlines_in_dataframe(out)
    if "rejected_type" in out.columns:
        out["rejected_type"] = out["rejected_type"].astype(str).str.strip().str.lower()
        out["subset_type"] = out["rejected_type"].map(SUBSET_TYPE_MAP).fillna(out["rejected_type"])
    return out


def dedupe_exact_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove only true duplicate pair rows.

    Rows are treated as duplicates only when the prompt, both candidate responses,
    and the subset label are identical. If the same prompt appears with different
    accepted or rejected responses, or appears once as `lin` and once as
    `too_risk`, all such rows are kept.
    """
    deduped = df.copy()
    deduped["_row_index"] = deduped.index
    subset_cols = ["prompt_text", "chosen_full", "rejected_full"]
    if "rejected_type" in deduped.columns:
        subset_cols.append("rejected_type")
    deduped = deduped.drop_duplicates(subset=subset_cols, keep="first")
    deduped = deduped.sort_values("_row_index").reset_index(drop=True)
    return deduped


def alternate_by_subset_type(df: pd.DataFrame) -> pd.DataFrame:
    """Interleave rebels_only and steals_only rows as much as possible, preserving within-type order."""
    rebels_rows = df[df["subset_type"] == "rebels_only"].copy()
    steals_rows = df[df["subset_type"] == "steals_only"].copy()

    first_type = None
    if not df.empty:
        first_type = str(df.iloc[0]["subset_type"])
    if first_type not in {"rebels_only", "steals_only"}:
        first_type = "rebels_only"

    queues: Dict[str, List[dict]] = {
        "rebels_only": rebels_rows.to_dict("records"),
        "steals_only": steals_rows.to_dict("records"),
    }

    order = [first_type, "steals_only" if first_type == "rebels_only" else "rebels_only"]
    combined: List[dict] = []
    next_idx = 0

    while queues["rebels_only"] and queues["steals_only"]:
        current_type = order[next_idx % 2]
        combined.append(queues[current_type].pop(0))
        next_idx += 1

    remainder_type = "rebels_only" if queues["rebels_only"] else "steals_only"
    combined.extend(queues[remainder_type])
    return pd.DataFrame(combined)


def limit_subset(df: pd.DataFrame, subset_type: str, max_rows: int) -> pd.DataFrame:
    """Keep up to max_rows from one subset, preserving existing order."""
    subset = df[df["subset_type"] == subset_type].copy().reset_index(drop=True)
    if max_rows is None:
        return subset
    return subset.head(max_rows).reset_index(drop=True)


def replace_select_wording(prompt_text: str, verb: str) -> str:
    """Replace whole-word 'select' occurrences with a chosen verb."""
    if verb == "select":
        return prompt_text
    return re.sub(r"\bselect\b", verb, prompt_text)


def apply_prompt_choice_verb_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Spread select/choose/pick across unique prompts in stable round-robin order."""
    out = df.copy()
    prompt_to_verb: Dict[str, str] = {}
    for idx, prompt_text in enumerate(out["prompt_text"].astype(str).drop_duplicates().tolist()):
        prompt_to_verb[prompt_text] = CHOICE_VERBS[idx % len(CHOICE_VERBS)]
    out["prompt_text"] = out["prompt_text"].astype(str).map(
        lambda prompt_text: replace_select_wording(prompt_text, prompt_to_verb[prompt_text])
    )
    return out


def write_dataset(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Raw held-out pairwise preference CSV")
    parser.add_argument(
        "--output_combined_csv",
        required=True,
        help="Combined output CSV after limiting to the requested rebels_only / steals_only target sizes",
    )
    parser.add_argument("--output_rebels_csv", required=True, help="Current rebels_only output CSV")
    parser.add_argument("--output_steals_csv", required=True, help="Current steals_only output CSV")
    parser.add_argument("--max_rebels_only", type=int, default=500, help="Maximum rebels_only rows (default: 500)")
    parser.add_argument("--max_steals_only", type=int, default=500, help="Maximum steals_only rows (default: 500)")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input_csv)
    normalized = normalize_reward_df(raw_df)
    deduped = dedupe_exact_pair_rows(normalized)
    full_combined = alternate_by_subset_type(deduped)
    full_rebels = full_combined[full_combined["subset_type"] == "rebels_only"].reset_index(drop=True)
    full_steals = full_combined[full_combined["subset_type"] == "steals_only"].reset_index(drop=True)

    rebels_only = limit_subset(full_rebels, "rebels_only", args.max_rebels_only)
    steals_only = limit_subset(full_steals, "steals_only", args.max_steals_only)
    combined = alternate_by_subset_type(
        pd.concat([rebels_only, steals_only], ignore_index=True)
    ).reset_index(drop=True)

    combined = apply_prompt_choice_verb_mix(combined)
    rebels_only = apply_prompt_choice_verb_mix(rebels_only)
    steals_only = apply_prompt_choice_verb_mix(steals_only)

    write_dataset(combined, Path(args.output_combined_csv))
    write_dataset(rebels_only, Path(args.output_rebels_csv))
    write_dataset(steals_only, Path(args.output_steals_csv))

    print(f"Raw rows: {len(normalized)}")
    print(f"Rows after exact pair dedupe: {len(deduped)}")
    print(f"Combined rows written: {len(combined)}")
    print(f"Rebels-only rows written: {len(rebels_only)}")
    print(f"Steals-only rows written: {len(steals_only)}")
    print(f"Available rebels_only rows before limiting: {len(full_rebels)}")
    print(f"Available steals_only rows before limiting: {len(full_steals)}")


if __name__ == "__main__":
    main()
