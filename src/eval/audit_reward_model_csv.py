#!/usr/bin/env python3
"""Audit reward-model CoT CSVs for prompt-meta leakage and label consistency."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from cot_csv_utils import PROMPT_META_REFERENCE_RE


VERBAL_PROB_MAP = {
    "Certainly will give": 1.00,
    "Almost certain to give": 0.95,
    "Highly likely to give": 0.90,
    "Very likely to give": 0.85,
    "Likely to give": 0.80,
    "Probable to give": 0.75,
    "Fairly likely to give": 0.70,
    "Somewhat likely to give": 0.65,
    "More than a half-chance of": 0.60,
    "Just more than a half-chance of": 0.55,
    "A half-chance of": 0.50,
    "Just less than a half-chance of": 0.45,
    "Less than a half-chance of": 0.40,
    "Somewhat unlikely to give": 0.35,
    "Fairly unlikely to give": 0.30,
    "Improbable to give": 0.25,
    "Unlikely to give": 0.20,
    "Very unlikely to give": 0.15,
    "Highly unlikely to give": 0.10,
    "Almost certainly won't give": 0.05,
    "Certainly won't give": 0.00,
}
VERBAL_PATTERNS = sorted(VERBAL_PROB_MAP, key=len, reverse=True)
AMOUNT_PATTERN = r"-?\$[\d,]+(?:\.\d+)?"
NUMERIC_OUTCOME_RE = re.compile(
    rf"(?:"
    rf"(?P<prob_pct_or_decimal_before_probability>\d*\.\d+|\d+(?:\.\d+)?%?)\s+probability\s+of|"
    rf"probability\s+(?P<prob_after_probability>\d*\.\d+|\d+(?:\.\d+)?%?)\s+of|"
    rf"(?P<prob_before_chance>\d*\.\d+|\d+(?:\.\d+)?%?)\s+chance\s+of|"
    rf"(?P<prob_before_of>\d*\.\d+|\d+(?:\.\d+)?%?)\s+of"
    rf")\s+(?P<amt>{AMOUNT_PATTERN})",
    re.IGNORECASE,
)
OPTIONS_BLOCK_RE = re.compile(
    r"following options[.:]\s*(.*?)\s*(?:You are completely free|You have complete freedom)",
    re.IGNORECASE | re.DOTALL,
)
OPTION_LINE_RE = re.compile(r"^[\(\[]?([A-Za-z]|\d{1,2})[\)\]]?[\.:]?\s+(.*)$")
ANSWER_JSON_RE = re.compile(r'\{\s*"answer"\s*:\s*"?(?P<answer>[^"}\s]+)"?\s*\}')


def normalize_option_label(label: Any) -> Optional[str]:
    text = str(label).strip().lower()
    if text.isdigit():
        return text
    if len(text) == 1 and "a" <= text <= "z":
        return text
    return None


def parse_probability_text(probability_text: str) -> float:
    if probability_text.endswith("%"):
        return float(probability_text[:-1]) / 100.0
    probability = float(probability_text)
    if probability > 1.0:
        return probability / 100.0
    return probability


def parse_ok_flag(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return bool(value)


def extract_answer_json_label(response_text: Any) -> Optional[str]:
    match = ANSWER_JSON_RE.search(str(response_text).replace("\\n", "\n"))
    if not match:
        return None
    return normalize_option_label(match.group("answer"))


def extract_prompt_options(prompt_text: str) -> Optional[List[Tuple[str, List[Tuple[float, float]]]]]:
    """Parse a prompt into option labels and (probability, amount) outcomes."""
    text = prompt_text.replace("\\n", "\n")
    block_match = OPTIONS_BLOCK_RE.search(text)
    if not block_match:
        return None

    option_lines = [line.strip() for line in block_match.group(1).strip().splitlines() if line.strip()]
    options: List[Tuple[str, List[Tuple[float, float]]]] = []
    for line in option_lines:
        line_match = OPTION_LINE_RE.match(line)
        if not line_match:
            continue
        label = line_match.group(1).lower()
        content = line_match.group(2)
        matches: List[Tuple[int, int, float, float]] = []

        for verbal in VERBAL_PATTERNS:
            verbal_re = re.compile(rf"{re.escape(verbal)}\s+({AMOUNT_PATTERN})", re.IGNORECASE)
            for match in verbal_re.finditer(content):
                amount = float(match.group(1).replace("$", "").replace(",", ""))
                matches.append((match.start(), match.end(), VERBAL_PROB_MAP[verbal], amount))

        for match in NUMERIC_OUTCOME_RE.finditer(content):
            probability_text = next(group for group in match.groups()[:-1] if group is not None)
            probability = parse_probability_text(probability_text)
            amount = float(match.group("amt").replace("$", "").replace(",", ""))
            matches.append((match.start(), match.end(), probability, amount))

        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        kept: List[Tuple[float, float]] = []
        used_ranges: List[Tuple[int, int]] = []
        for start, end, probability, amount in matches:
            if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
                continue
            used_ranges.append((start, end))
            kept.append((probability, amount))

        if kept:
            options.append((label, kept))
    return options or None


def logsumexp(values: List[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def cara_rank_score(outcomes: List[Tuple[float, float]]) -> float:
    """Lower is better. Equivalent to maximizing CARA expected utility with alpha=0.01."""
    terms = []
    for probability, amount in outcomes:
        if probability <= 0:
            continue
        terms.append(math.log(probability) - 0.01 * amount)
    return logsumexp(terms)


def audit_reward_model_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Return row-level and aggregate audit results for a reward-model CoT CSV."""
    rows: List[Dict[str, Any]] = []
    prompt_meta_reference_rows: List[str] = []
    parse_failure_rows: List[str] = []
    chosen_label_mismatch_rows: List[str] = []
    chosen_response_mismatch_rows: List[str] = []
    rejected_response_mismatch_rows: List[str] = []

    for row_index, row in enumerate(df.to_dict(orient="records"), start=1):
        situation_id = str(row.get("situation_id", row_index))
        prompt_text = str(row.get("prompt_text", ""))
        chosen_expected = normalize_option_label(row.get("chosen_expected"))
        rejected_expected = normalize_option_label(row.get("rejected_expected"))
        chosen_answer = normalize_option_label(row.get("chosen_answer"))
        rejected_answer = normalize_option_label(row.get("rejected_answer"))
        chosen_answer_json = extract_answer_json_label(row.get("chosen_full", ""))
        rejected_answer_json = extract_answer_json_label(row.get("rejected_full", ""))
        has_prompt_meta_reference = any(
            PROMPT_META_REFERENCE_RE.search(str(row.get(column_name, "")))
            for column_name in ("chosen_full", "rejected_full")
        )
        parsed_options = extract_prompt_options(prompt_text)
        parse_status = "ok"
        best_labels: List[str] = []
        best_scores: List[Tuple[str, float]] = []
        chosen_matches_cara = None
        chosen_response_matches_expected = (
            chosen_expected is not None
            and chosen_answer == chosen_expected
            and chosen_answer_json == chosen_expected
            and parse_ok_flag(row.get("chosen_ok", True))
        )
        rejected_response_matches_expected = (
            rejected_expected is not None
            and rejected_answer == rejected_expected
            and rejected_answer_json == rejected_expected
            and parse_ok_flag(row.get("rejected_ok", True))
        )

        if parsed_options is None:
            parse_status = "parse_failed"
            parse_failure_rows.append(situation_id)
        else:
            best_scores = [(label, cara_rank_score(outcomes)) for label, outcomes in parsed_options]
            best_score = min(score for _, score in best_scores)
            best_labels = sorted(
                label for label, score in best_scores if abs(score - best_score) < 1e-10
            )
            if chosen_expected is None:
                parse_status = "bad_chosen_label"
                parse_failure_rows.append(situation_id)
            else:
                chosen_matches_cara = chosen_expected in set(best_labels)
                if not chosen_matches_cara:
                    chosen_label_mismatch_rows.append(situation_id)

        if has_prompt_meta_reference:
            prompt_meta_reference_rows.append(situation_id)
        if not chosen_response_matches_expected:
            chosen_response_mismatch_rows.append(situation_id)
        if not rejected_response_matches_expected:
            rejected_response_mismatch_rows.append(situation_id)

        excluded_reason_codes: List[str] = []
        if has_prompt_meta_reference:
            excluded_reason_codes.append("prompt_meta_reference")
        if parse_status != "ok":
            excluded_reason_codes.append(parse_status)
        if chosen_matches_cara is False:
            excluded_reason_codes.append("chosen_label_mismatch")
        if not chosen_response_matches_expected:
            excluded_reason_codes.append("chosen_response_mismatch")
        if not rejected_response_matches_expected:
            excluded_reason_codes.append("rejected_response_mismatch")

        rows.append(
            {
                "row_index": row_index,
                "situation_id": situation_id,
                "has_prompt_meta_reference": has_prompt_meta_reference,
                "parse_status": parse_status,
                "chosen_expected": chosen_expected,
                "rejected_expected": rejected_expected,
                "chosen_answer": chosen_answer,
                "rejected_answer": rejected_answer,
                "chosen_answer_json": chosen_answer_json,
                "rejected_answer_json": rejected_answer_json,
                "computed_best_labels": best_labels,
                "chosen_matches_cara": chosen_matches_cara,
                "chosen_response_matches_expected": chosen_response_matches_expected,
                "rejected_response_matches_expected": rejected_response_matches_expected,
                "excluded_reason_codes": excluded_reason_codes,
                "option_scores": [
                    {"label": label, "cara_rank_score": score} for label, score in best_scores
                ],
            }
        )

    excluded_rows = {
        record["situation_id"]
        for record in rows
        if record["has_prompt_meta_reference"]
        or record["parse_status"] != "ok"
        or record["chosen_matches_cara"] is False
        or not record["chosen_response_matches_expected"]
        or not record["rejected_response_matches_expected"]
    }
    kept_rows = [record["situation_id"] for record in rows if record["situation_id"] not in excluded_rows]
    return {
        "num_rows": len(rows),
        "rows": rows,
        "prompt_meta_reference_rows": sorted(set(prompt_meta_reference_rows), key=int),
        "parse_failure_rows": sorted(set(parse_failure_rows), key=int),
        "chosen_label_mismatch_rows": sorted(set(chosen_label_mismatch_rows), key=int),
        "chosen_response_mismatch_rows": sorted(set(chosen_response_mismatch_rows), key=int),
        "rejected_response_mismatch_rows": sorted(set(rejected_response_mismatch_rows), key=int),
        "excluded_rows": sorted(excluded_rows, key=int),
        "kept_rows": kept_rows,
    }


def dedupe_by_situation_id(df: pd.DataFrame) -> pd.DataFrame:
    if "situation_id" not in df.columns:
        return df.copy()
    out = df.copy()
    out["_audit_original_row_order"] = range(len(out))
    out = out.drop_duplicates(subset=["situation_id"], keep="first")
    out = out.sort_values("_audit_original_row_order").drop(columns=["_audit_original_row_order"])
    return out.reset_index(drop=True)


def build_audit_report(
    path: Path,
    audit: Dict[str, Any],
    clean_csv_path: Optional[Path],
    kept_rows_written: Optional[int],
    kept_unique_situations: Optional[int],
    deduped_by_situation_id: bool,
) -> Dict[str, Any]:
    return {
        "source_csv": str(path),
        "clean_csv": None if clean_csv_path is None else str(clean_csv_path),
        "num_rows": audit["num_rows"],
        "num_prompt_meta_reference_rows": len(audit["prompt_meta_reference_rows"]),
        "num_parse_failure_rows": len(audit["parse_failure_rows"]),
        "num_chosen_label_mismatch_rows": len(audit["chosen_label_mismatch_rows"]),
        "num_chosen_response_mismatch_rows": len(audit["chosen_response_mismatch_rows"]),
        "num_rejected_response_mismatch_rows": len(audit["rejected_response_mismatch_rows"]),
        "num_excluded_rows": len(audit["excluded_rows"]),
        "num_kept_rows": len(audit["kept_rows"]),
        "num_kept_rows_written": kept_rows_written,
        "num_kept_unique_situations": kept_unique_situations,
        "deduped_by_situation_id": deduped_by_situation_id,
        "prompt_meta_reference_rows": audit["prompt_meta_reference_rows"],
        "parse_failure_rows": audit["parse_failure_rows"],
        "chosen_label_mismatch_rows": audit["chosen_label_mismatch_rows"],
        "chosen_response_mismatch_rows": audit["chosen_response_mismatch_rows"],
        "rejected_response_mismatch_rows": audit["rejected_response_mismatch_rows"],
        "excluded_rows": audit["excluded_rows"],
        "kept_rows": audit["kept_rows"],
        "row_details": audit["rows"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Reward-model CSV to audit")
    parser.add_argument("--write-clean-csv", help="Optional output path for the filtered clean CSV")
    parser.add_argument("--write-report-json", help="Optional output path for the audit report JSON")
    parser.add_argument(
        "--dedupe-by-situation-id",
        action="store_true",
        help="When writing a clean CSV, keep only the first passing row for each situation_id.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    df = pd.read_csv(csv_path)
    audit = audit_reward_model_dataframe(df)

    clean_csv_path = None if not args.write_clean_csv else Path(args.write_clean_csv).expanduser().resolve()
    kept_rows_written = None
    kept_unique_situations = None
    if clean_csv_path is not None:
        keep_ids = set(audit["kept_rows"])
        clean_df = df[df["situation_id"].astype(str).isin(keep_ids)].copy()
        kept_unique_situations = clean_df["situation_id"].astype(str).nunique() if "situation_id" in clean_df.columns else None
        if args.dedupe_by_situation_id:
            clean_df = dedupe_by_situation_id(clean_df)
        kept_rows_written = len(clean_df)
        clean_df.to_csv(clean_csv_path, index=False)

    report = build_audit_report(
        csv_path,
        audit,
        clean_csv_path,
        kept_rows_written,
        kept_unique_situations,
        args.dedupe_by_situation_id,
    )
    if args.write_report_json:
        report_path = Path(args.write_report_json).expanduser().resolve()
        report_path.write_text(json.dumps(report, indent=2))

    print(f"{csv_path}")
    print(f"  rows: {audit['num_rows']}")
    print(f"  prompt-meta reference rows: {len(audit['prompt_meta_reference_rows'])}")
    print(f"  parse-failure rows: {len(audit['parse_failure_rows'])}")
    print(f"  chosen-label mismatch rows: {len(audit['chosen_label_mismatch_rows'])}")
    print(f"  chosen-response mismatch rows: {len(audit['chosen_response_mismatch_rows'])}")
    print(f"  rejected-response mismatch rows: {len(audit['rejected_response_mismatch_rows'])}")
    print(f"  excluded rows: {len(audit['excluded_rows'])}")
    print(f"  kept rows: {len(audit['kept_rows'])}")
    if clean_csv_path is not None:
        print(f"  wrote clean CSV: {clean_csv_path}")
        if args.dedupe_by_situation_id:
            print(f"  kept rows written after situation dedupe: {kept_rows_written}")
        if kept_unique_situations is not None:
            print(f"  kept unique situations before optional dedupe: {kept_unique_situations}")
    if args.write_report_json:
        print(f"  wrote report JSON: {Path(args.write_report_json).expanduser().resolve()}")


if __name__ == "__main__":
    main()
