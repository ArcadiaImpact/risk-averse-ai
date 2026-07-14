#!/usr/bin/env python3
"""Build interleaved transfer-to-other-quantities benchmark CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "Transfer-to-other-quantities tests"
DEFAULT_REPO_OUTPUT_DIR = SCRIPT_DIR / "data" / "transfer_to_other_quantities"
DEFAULT_PROJECT_OUTPUT_DIR = DEFAULT_SOURCE_DIR

STAKE_ORDER = (
    ("low_stakes_training", "low_stakes_training_set"),
    ("medium_stakes_validation", "medium_stakes_val_set"),
    ("high_stakes_test", "high_stakes_test_set"),
    ("astronomical_stakes_deployment", "astronomical_stakes_deployment_set"),
)
CONDITIONS = ("gpu_hours", "lives_saved", "money_for_user")


def build_condition_dataframe(source_dir: Path, condition: str) -> pd.DataFrame:
    per_stake_rows = []
    expected_situation_count = None

    for source_stakes, filename_fragment in STAKE_ORDER:
        source_filename = f"2026_04_11_{condition}_{filename_fragment}_gambles.csv"
        source_path = source_dir / source_filename
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source CSV: {source_path}")

        df = pd.read_csv(source_path)
        ordered_situation_ids = list(dict.fromkeys(df["situation_id"].tolist()))
        if expected_situation_count is None:
            expected_situation_count = len(ordered_situation_ids)
        elif len(ordered_situation_ids) != expected_situation_count:
            raise ValueError(
                f"{condition}: expected {expected_situation_count} situations per stakes file, "
                f"but {source_filename} has {len(ordered_situation_ids)}"
            )

        per_stake_rows.append(
            {
                "source_stakes": source_stakes,
                "source_filename": source_filename,
                "dataframe": df,
                "ordered_situation_ids": ordered_situation_ids,
            }
        )

    interleaved_blocks = []
    next_situation_id = 0
    for situation_offset in range(expected_situation_count or 0):
        for stake_block in per_stake_rows:
            source_situation_id = stake_block["ordered_situation_ids"][situation_offset]
            block = stake_block["dataframe"][stake_block["dataframe"]["situation_id"] == source_situation_id].copy()
            block["source_stakes"] = stake_block["source_stakes"]
            block["source_condition"] = condition
            block["source_csv_name"] = stake_block["source_filename"]
            block["source_situation_id"] = int(source_situation_id)
            block["situation_id"] = next_situation_id
            interleaved_blocks.append(block)
            next_situation_id += 1

    combined = pd.concat(interleaved_blocks, ignore_index=True)
    metadata_columns = [
        "source_stakes",
        "source_condition",
        "source_csv_name",
        "source_situation_id",
    ]
    ordered_columns = []
    for column in combined.columns:
        if column == "situation_id":
            ordered_columns.append(column)
            ordered_columns.extend([meta for meta in metadata_columns if meta in combined.columns])
        elif column not in metadata_columns:
            ordered_columns.append(column)
    return combined[ordered_columns]


def write_condition_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--repo_output_dir", type=Path, default=DEFAULT_REPO_OUTPUT_DIR)
    parser.add_argument(
        "--mirror_to_project_dir",
        action="store_true",
        help="Also write copies into the project-level Transfer-to-other-quantities tests folder.",
    )
    parser.add_argument("--project_output_dir", type=Path, default=DEFAULT_PROJECT_OUTPUT_DIR)
    args = parser.parse_args()

    for condition in CONDITIONS:
        combined = build_condition_dataframe(args.source_dir, condition)
        output_name = f"2026_04_11_{condition}_transfer_benchmark_interleaved_1000_situations.csv"
        write_condition_dataframe(combined, args.repo_output_dir / output_name)
        if args.mirror_to_project_dir:
            write_condition_dataframe(combined, args.project_output_dir / output_name)
        print(
            f"{condition}: wrote {combined['situation_id'].nunique()} situations / {len(combined)} rows "
            f"to {args.repo_output_dir / output_name}"
        )


if __name__ == "__main__":
    main()
