"""Dataset registry + situation loading for the risk-averse benchmark eval.

This is the "datasets" module of the library split (named ``situations`` to
avoid shadowing HuggingFace's top-level ``datasets`` package, since the eval
dir goes on ``sys.path``). It holds:

* the built-in dataset alias tables and variant resolution
  (``resolve_builtin_dataset_path`` and friends);
* the schema constants and column validation;
* ``build_situations`` — group option-level CSV rows into situation objects
  with the exact CARA/linear/EV labels the scoring paths consume;
* the situation manifest helpers and ``build_eval_prompt``.

All code here is moved verbatim from the original ``evaluate.py`` monolith (no
behavioral change — this is a measurement instrument) and is torch-free, so the
async ``run_evaluation`` path imports it without pulling in torch.
"""
import ast
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from answer_parser import infer_option_label_style

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL_TEMPERATURE = 0.6
CANONICAL_DATASET_ALIASES = {
    "low_stakes_training": "data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv",
    "medium_stakes_validation": "data/2026_03_22_medium_stakes_val_set_500_Rebels.csv",
    "high_stakes_test": "data/2026_03_22_high_stakes_test_set_1000_Rebels.csv",
    "astronomical_stakes_deployment": "data/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv",
    "steals_test": "data/2026_03_22_test_set_1000_Steals.csv",
}
CURRENT_EXTRA_DATASET_ALIASES = {
    "low_stakes_validation": "data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv",
    "low_stakes_training_lin_only": "data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv",
    "low_stakes_validation_lin_only": "data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv",
    "medium_stakes_validation_rebels_only": "data/2026_03_22_medium_stakes_val_set_500_Rebels.csv",
    "high_stakes_test_rebels_only": "data/2026_03_22_high_stakes_test_set_1000_Rebels.csv",
    "astronomical_stakes_deployment_rebels_only": "data/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv",
    "gpu_hours_transfer_benchmark": "data/transfer_to_other_quantities/2026_04_11_gpu_hours_transfer_benchmark_interleaved_1000_situations.csv",
    "lives_saved_transfer_benchmark": "data/transfer_to_other_quantities/2026_04_11_lives_saved_transfer_benchmark_interleaved_1000_situations.csv",
    "money_for_user_transfer_benchmark": "data/transfer_to_other_quantities/2026_04_11_money_for_user_transfer_benchmark_interleaved_1000_situations.csv",
}
EXTRA_DATASET_ALIASES = CURRENT_EXTRA_DATASET_ALIASES
_RESOLVABLE_DATASET_ALIASES = {
    **CANONICAL_DATASET_ALIASES,
    **EXTRA_DATASET_ALIASES,
}
DATASET_ALIASES = dict(_RESOLVABLE_DATASET_ALIASES)
DATASET_VARIANT_PATHS = {
    "medium_stakes_validation": {
        "rebels_only": "data/2026_03_22_medium_stakes_val_set_500_Rebels.csv",
    },
    "high_stakes_test": {
        "rebels_only": "data/2026_03_22_high_stakes_test_set_1000_Rebels.csv",
        "steals_only": "data/2026_03_22_test_set_1000_Steals.csv",
    },
    "astronomical_stakes_deployment": {
        "rebels_only": "data/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv",
        "steals_only": "data/2026_03_22_test_set_1000_Steals.csv",
    },
}
DATASET_ALIAS_BASE_NAMES = {
    "medium_stakes_validation": "medium_stakes_validation",
    "medium_stakes_validation_rebels_only": "medium_stakes_validation",
    "high_stakes_test": "high_stakes_test",
    "high_stakes_test_rebels_only": "high_stakes_test",
    "astronomical_stakes_deployment": "astronomical_stakes_deployment",
    "astronomical_stakes_deployment_rebels_only": "astronomical_stakes_deployment",
    "steals_test": "steals_test",
    "gpu_hours_transfer_benchmark": "gpu_hours_transfer_benchmark",
    "lives_saved_transfer_benchmark": "lives_saved_transfer_benchmark",
    "money_for_user_transfer_benchmark": "money_for_user_transfer_benchmark",
}
DATASET_ALIAS_VARIANTS = {
    "medium_stakes_validation": "rebels_only",
    "medium_stakes_validation_rebels_only": "rebels_only",
    "high_stakes_test": "rebels_only",
    "high_stakes_test_rebels_only": "rebels_only",
    "astronomical_stakes_deployment": "rebels_only",
    "astronomical_stakes_deployment_rebels_only": "rebels_only",
    "steals_test": "steals_only",
    "gpu_hours_transfer_benchmark": "default",
    "lives_saved_transfer_benchmark": "default",
    "money_for_user_transfer_benchmark": "default",
}
DATASET_VARIANT_SYNONYMS = {
    "default": "default",
    "rebels_only": "rebels_only",
    "rebels": "rebels_only",
    "rebel_cooperate": "rebels_only",
    "rebel": "rebels_only",
    "with_steals": "steals_only",
    "steals_only": "steals_only",
    "steals": "steals_only",
    "steal_only": "steals_only",
    "steal_mixed": "steals_only",
    "combined": "combined",
    "unified": "combined",
}
DEFAULT_NUM_SITUATIONS_BY_DATASET = {
    "low_stakes_training": 200,
    "low_stakes_validation": 200,
    "low_stakes_training_lin_only": 200,
    "low_stakes_validation_lin_only": 200,
    "medium_stakes_validation": 200,
    "medium_stakes_validation_rebels_only": 200,
    "medium_stakes_validation_steals_only": 500,
    "high_stakes_test": 1000,
    "high_stakes_test_rebels_only": 1000,
    "astronomical_stakes_deployment": 1000,
    "astronomical_stakes_deployment_rebels_only": 1000,
    "steals_test": 1000,
    "gpu_hours_transfer_benchmark": 1000,
    "lives_saved_transfer_benchmark": 1000,
    "money_for_user_transfer_benchmark": 1000,
}
REQUIRED_COLUMNS = {"situation_id", "prompt_text", "option_index", "option_type"}
CARA_COLUMNS = {"is_best_cara_display", "CARA_correct_labels", "CARA_alpha_0_01_best_labels"}
LINEAR_COLUMNS = {"is_best_linear_display", "linear_correct_labels", "linear_best_labels"}
PREFERRED_CARA_LABEL_COLUMNS = ("CARA_correct_labels", "CARA_alpha_0_01_best_labels")
PREFERRED_LINEAR_LABEL_COLUMNS = ("linear_correct_labels", "linear_best_labels")
LIN_ONLY_BUCKET_LABELS = {"lin_only", "linear_only"}
SUBSET_TYPES = ("rebels_only", "steals_only")
PROBABILITY_FORMATS = ("numerical", "verbal")
SOURCE_STAKES = (
    "low_stakes_training",
    "medium_stakes_validation",
    "high_stakes_test",
    "astronomical_stakes_deployment",
)
BEHAVIORAL_OPTION_TYPES = {"Cooperate", "Rebel", "Steal"}


def resolve_path(path):
    """Resolve a path relative to either this script or the current working directory."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    script_relative = os.path.abspath(os.path.join(SCRIPT_DIR, expanded))
    if os.path.exists(script_relative):
        return script_relative
    return os.path.abspath(expanded)


def normalize_dataset_variant(dataset_variant: str) -> str:
    """Normalize user-facing dataset variant names."""
    normalized = str(dataset_variant).strip().lower()
    if normalized not in DATASET_VARIANT_SYNONYMS:
        raise ValueError(
            "Unsupported --dataset_variant. Choose one of: "
            + ", ".join(sorted(DATASET_VARIANT_SYNONYMS))
        )
    return DATASET_VARIANT_SYNONYMS[normalized]


def resolve_default_num_situations(args) -> Optional[int]:
    """Return the recommended default situation count for the selected dataset."""
    if args.dataset in DEFAULT_NUM_SITUATIONS_BY_DATASET:
        return DEFAULT_NUM_SITUATIONS_BY_DATASET[args.dataset]
    if args.dataset_base_alias == "medium_stakes_validation":
        return 200
    if args.dataset_base_alias in {"high_stakes_test", "astronomical_stakes_deployment"}:
        if args.resolved_dataset_variant in {"rebels_only", "steals_only"}:
            return 1000
    return None


def resolve_builtin_dataset_path(dataset_name: str, dataset_variant: str):
    """Resolve built-in dataset alias plus optional variant override to a CSV path."""
    normalized_variant = normalize_dataset_variant(dataset_variant)
    base_dataset = DATASET_ALIAS_BASE_NAMES.get(dataset_name, dataset_name)

    if normalized_variant == "default":
        return resolve_path(DATASET_ALIASES[dataset_name]), DATASET_ALIAS_VARIANTS.get(dataset_name, "default"), base_dataset

    variant_paths = DATASET_VARIANT_PATHS.get(base_dataset)
    if variant_paths is None:
        raise ValueError(
            f"--dataset_variant {normalized_variant!r} is not supported for dataset {dataset_name!r}."
        )
    if normalized_variant not in variant_paths:
        available = ", ".join(sorted(variant_paths))
        raise ValueError(
            f"Built-in dataset variant {normalized_variant!r} is not configured yet for {base_dataset!r}. "
            f"Available built-in variants: {available}."
        )
    return resolve_path(variant_paths[normalized_variant]), normalized_variant, base_dataset


def validate_dataset_columns(df, dataset_path):
    """Validate that the dataset has the minimum schema needed for evaluation."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Dataset path: {dataset_path}"
        )

    if not any(col in df.columns for col in CARA_COLUMNS):
        raise ValueError(
            "Dataset is missing CARA-label columns. Expected at least one of "
            f"{sorted(CARA_COLUMNS)}\nDataset path: {dataset_path}"
        )


def option_numbers_from_label_columns(sit_data: pd.DataFrame, column_names) -> set:
    """Parse 1-based option numbers from the first available label-list column."""
    for column_name in column_names:
        if column_name not in sit_data.columns:
            continue
        labels = parse_label_list(sit_data[column_name].iloc[0])
        option_numbers = {
            label_to_option_number(label)
            for label in labels
            if label_to_option_number(label) is not None
        }
        if option_numbers:
            return option_numbers
    return set()


def remove_instruction_suffix(prompt):
    """Remove the instruction about how to respond from the end of the prompt."""
    patterns = [
        r"\s*You can think before answering,.*?would select\.",
        r"\s*You can think.*?must finish with.*?\.",
    ]
    for pattern in patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE | re.DOTALL)
    return prompt.strip()


def clean_bucket_label(value):
    """Normalize low_bucket_label strings like '"lin_only"' -> 'lin_only'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    return s.lower()


def is_lin_only_label(bucket_label: Optional[str]) -> bool:
    """Return True when a bucket label indicates LIN-only situations."""
    if bucket_label is None:
        return False
    return clean_bucket_label(bucket_label) in LIN_ONLY_BUCKET_LABELS


def is_lin_only_situation(linear_best: set, cara_best: set, bucket_label: Optional[str]) -> bool:
    """Detect LIN-only situations using labels and fallback set disagreement."""
    if is_lin_only_label(bucket_label):
        return True
    return bool(linear_best and cara_best and linear_best != cara_best)


def parse_label_list(value):
    """Parse list-like label fields stored as JSON strings in CSV."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value).strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            return [parsed]
        return [str(parsed)]
    except Exception:
        s = s.strip('"').strip("'")
        if not s:
            return []
        if "," in s:
            return [part.strip().strip('"').strip("'") for part in s.split(",") if part.strip()]
        return [s]


def parse_literal_list(value):
    """Parse a Python/JSON-style list cell such as '[1, 2]'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def infer_label_style_from_allowed_labels(value) -> Optional[str]:
    """Infer answer label style directly from the stored allowed_labels column."""
    labels = parse_label_list(value)
    if not labels:
        return None
    first = str(labels[0]).strip()
    if not first:
        return None
    if first.isalpha():
        return "letters"
    if first.isdigit():
        return "numbers"
    return None


def compute_expected_value_from_row(row: pd.Series) -> Optional[float]:
    """Compute exact EV from prizes_display and probs_percent when available."""
    if "prizes_display" not in row or "probs_percent" not in row:
        return None
    prizes = parse_literal_list(row.get("prizes_display"))
    probs_percent = parse_literal_list(row.get("probs_percent"))
    if not prizes or not probs_percent or len(prizes) != len(probs_percent):
        return None
    try:
        probs = [float(p) / 100.0 for p in probs_percent]
        prob_sum = sum(probs)
        if prob_sum > 0 and abs(prob_sum - 1.0) > 1e-9:
            probs = [p / prob_sum for p in probs]
        return float(sum(float(prize) * prob for prize, prob in zip(prizes, probs)))
    except Exception:
        return None


def parse_bool_like(value):
    """Parse bool-ish CSV values robustly (handles numpy/pandas/string forms)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return bool(value)


def infer_probability_format(prompt_text):
    """Best-effort fallback if explicit use_verbal_probs is missing."""
    if not isinstance(prompt_text, str):
        return None
    if re.search(r"\d+\s*%", prompt_text):
        return "numerical"
    verbal_markers = [
        "very likely",
        "likely",
        "unlikely",
        "very unlikely",
        "almost certain",
        "almost no chance",
        "small chance",
    ]
    prompt_lower = prompt_text.lower()
    if any(marker in prompt_lower for marker in verbal_markers):
        return "verbal"
    return None


def probability_format_from_value(use_verbal_probs_value, prompt_text=None):
    parsed_bool = parse_bool_like(use_verbal_probs_value)
    if parsed_bool is True:
        return "verbal"
    if parsed_bool is False:
        return "numerical"
    return infer_probability_format(prompt_text)


def infer_subset_type(raw_subset_type, option_types_besides_cooperate: List[str]) -> str:
    """Normalize subset labels, inferring them from option types if needed."""
    if raw_subset_type is not None and not (isinstance(raw_subset_type, float) and pd.isna(raw_subset_type)):
        subset_type = str(raw_subset_type).strip().lower().replace("-", "_")
        if subset_type in {"rebels_only", "rebel_cooperate"}:
            return "rebels_only"
        if subset_type in {"steals_only", "steal_mixed", "with_steals"}:
            return "steals_only"
    if "steal" in option_types_besides_cooperate:
        return "steals_only"
    return "rebels_only"


def extract_situation_manifest_entry(situation: Dict) -> Dict:
    """Return compact per-situation metadata for ordering and subgroup summaries."""
    return {
        "situation_id": situation["situation_id"],
        "dataset_position": situation.get("dataset_position"),
        "subset_type": situation.get("subset_type"),
        "source_stakes": situation.get("source_stakes"),
        "source_condition": situation.get("source_condition"),
        "option_types_besides_cooperate": situation.get("option_types_besides_cooperate"),
        "num_options": situation.get("num_options"),
        "probability_format": situation.get("probability_format"),
    }


def build_situation_manifest(situations: List[Dict]) -> List[Dict]:
    """Build ordered situation metadata for the selected evaluation slice."""
    return [extract_situation_manifest_entry(sit) for sit in situations]


def build_situation_manifest_index(situations: List[Dict]) -> Dict[int, Dict]:
    """Index selected situations by situation_id for metadata backfilling."""
    return {entry["situation_id"]: entry for entry in build_situation_manifest(situations)}


def annotate_rows_with_situation_metadata(rows: List[Dict], situation_index: Dict[int, Dict]):
    """Backfill per-situation metadata onto result-like rows, including resumed checkpoints."""
    for row in rows:
        sid = row.get("situation_id")
        if sid is None:
            continue
        manifest = situation_index.get(sid)
        if not manifest:
            continue
        for key, value in manifest.items():
            if key in {"subset_type", "option_types_besides_cooperate"}:
                row[key] = value
                continue
            if row.get(key) is None:
                row[key] = value


def label_to_option_number(label):
    """Convert a label like 'a' or '1' into a 1-based option number."""
    s = str(label).strip().lower()
    if s.isdigit():
        return int(s)
    if len(s) == 1 and "a" <= s <= "z":
        return ord(s) - ord("a") + 1
    return None


def build_situations(df: pd.DataFrame, num_situations: Optional[int]):
    """Group rows into situation objects with option metadata."""
    situations = []
    situation_ids = df["situation_id"].unique()
    if num_situations is not None:
        situation_ids = situation_ids[:num_situations]
    for dataset_position, sit_id in enumerate(situation_ids, start=1):
        sit_data = df[df["situation_id"] == sit_id]
        prompt_raw = sit_data["prompt_text"].iloc[0]
        num_options = len(sit_data)
        use_verbal_probs = sit_data["use_verbal_probs"].iloc[0] if "use_verbal_probs" in df.columns else None
        source_stakes = sit_data["source_stakes"].iloc[0] if "source_stakes" in df.columns else None
        source_condition = sit_data["source_condition"].iloc[0] if "source_condition" in df.columns else None
        source_csv_name = sit_data["source_csv_name"].iloc[0] if "source_csv_name" in df.columns else None
        source_situation_id = sit_data["source_situation_id"].iloc[0] if "source_situation_id" in df.columns else None
        low_bucket_label = (
            clean_bucket_label(sit_data["low_bucket_label"].iloc[0]) if "low_bucket_label" in df.columns else None
        )
        raw_subset_type = sit_data["subset_type"].iloc[0] if "subset_type" in df.columns else None
        option_types_besides_cooperate = sorted(
            {
                str(v).strip().lower()
                for v in sit_data["option_type"].dropna().tolist()
                if str(v).strip().lower() != "cooperate"
            }
        )
        subset_type = infer_subset_type(raw_subset_type, option_types_besides_cooperate)

        linear_best_indices_0 = set()
        linear_best_option_numbers = set()
        has_linear_info = False
        if "is_best_linear_display" in df.columns:
            has_linear_info = True
            linear_best_indices_0 = set(
                int(idx) for idx in sit_data.loc[sit_data["is_best_linear_display"] == True, "option_index"]
            )
            linear_best_option_numbers = {idx + 1 for idx in linear_best_indices_0}
        elif any(column in df.columns for column in PREFERRED_LINEAR_LABEL_COLUMNS):
            has_linear_info = True
            linear_best_option_numbers = option_numbers_from_label_columns(
                sit_data,
                PREFERRED_LINEAR_LABEL_COLUMNS,
            )
            linear_best_indices_0 = {n - 1 for n in linear_best_option_numbers}
        if not linear_best_option_numbers:
            has_linear_info = False

        cara001_best_option_numbers = option_numbers_from_label_columns(
            sit_data,
            PREFERRED_CARA_LABEL_COLUMNS,
        )

        if not cara001_best_option_numbers and "is_best_cara_display" in df.columns:
            cara001_best_option_numbers = {
                int(idx) + 1 for idx in sit_data.loc[sit_data["is_best_cara_display"] == True, "option_index"]
            }

        bucket_label = low_bucket_label
        if bucket_label is None and linear_best_option_numbers and cara001_best_option_numbers:
            if linear_best_option_numbers == cara001_best_option_numbers:
                bucket_label = "both"
        lin_only = is_lin_only_situation(linear_best_option_numbers, cara001_best_option_numbers, bucket_label)

        options = {}
        best_cara_indices = set()
        expected_values_by_index = {}
        for _, row in sit_data.iterrows():
            idx = int(row["option_index"])
            letter = chr(ord("a") + idx)
            number = str(idx + 1)
            is_best_cara = row.get("is_best_cara_display", False) == True
            if not is_best_cara and cara001_best_option_numbers:
                # Fallback for datasets that store only list-style CARA label columns.
                is_best_cara = (idx + 1) in cara001_best_option_numbers
            expected_value = compute_expected_value_from_row(row)
            if expected_value is not None:
                expected_values_by_index[idx] = expected_value
            option_data = {
                "type": row["option_type"],
                "is_best_cara": is_best_cara,
                "is_best_linear": (idx in linear_best_indices_0) if has_linear_info else None,
                "option_index": idx,
                "expected_value": expected_value,
                # Backward-compatible alias used by some downstream EV summaries.
                "eu_linear": expected_value,
            }
            options[letter] = option_data
            options[number] = option_data
            if is_best_cara:
                best_cara_indices.add(idx)

        max_expected_value = None
        min_expected_value = None
        best_expected_value_indices = set()
        worst_expected_value_indices = set()
        unique_option_data = {id(v): v for v in options.values()}.values()
        if expected_values_by_index:
            max_expected_value = max(expected_values_by_index.values())
            min_expected_value = min(expected_values_by_index.values())
            best_expected_value_indices = {
                idx for idx, value in expected_values_by_index.items() if abs(value - max_expected_value) < 1e-12
            }
            worst_expected_value_indices = {
                idx for idx, value in expected_values_by_index.items() if abs(value - min_expected_value) < 1e-12
            }
        for option_data in unique_option_data:
            idx = option_data["option_index"]
            option_data["is_worst_linear"] = idx in worst_expected_value_indices if expected_values_by_index else None

        situations.append(
            {
                "situation_id": sit_id,
                "dataset_position": dataset_position,
                "subset_type": subset_type,
                "option_types_besides_cooperate": option_types_besides_cooperate,
                "prompt_raw": prompt_raw,
                "num_options": num_options,
                "answer_label_style": (
                    infer_option_label_style(prompt_raw, num_options)
                    or infer_label_style_from_allowed_labels(
                        sit_data["allowed_labels"].iloc[0] if "allowed_labels" in df.columns else None
                    )
                ),
                "options": options,
                "probability_format": probability_format_from_value(use_verbal_probs, prompt_raw),
                "bucket_label": bucket_label,
                "is_lin_only": lin_only,
                "best_cara_indices": sorted(best_cara_indices),
                "source_stakes": source_stakes,
                "source_condition": source_condition,
                "source_csv_name": source_csv_name,
                "source_situation_id": source_situation_id,
                "max_expected_value": max_expected_value,
                "min_expected_value": min_expected_value,
                "best_expected_value_indices": sorted(best_expected_value_indices),
                "worst_expected_value_indices": sorted(worst_expected_value_indices),
            }
        )
    return situations


def filter_lin_only_situations(situations: List[Dict]) -> List[Dict]:
    """Keep only LIN-only situations where linear-best and CARA-best labels disagree."""
    return [sit for sit in situations if sit.get("is_lin_only")]


def build_eval_prompt(prompt_raw: str, prompt_suffix: str) -> str:
    """Normalize the dataset prompt and append an optional suffix."""
    prompt = remove_instruction_suffix(prompt_raw)
    return f"{prompt}\n\n{prompt_suffix}".strip() if prompt_suffix else prompt
