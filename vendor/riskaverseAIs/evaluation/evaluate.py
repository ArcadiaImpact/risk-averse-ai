#!/usr/bin/env python3
"""
Evaluate local HF/PEFT models on the risk-averse benchmark with permissive parsing.

Default behavior matches the original standard evaluator (single run, no steering).
Optional steering controls allow loading precomputed steering directions and
running alpha sweeps.
"""

import argparse
import ast
import gc
import json
import os
import re
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from answer_parser import infer_option_label_style, parse_choice_with_strategy
from dataset_schema_utils import ensure_option_level_dataframe
from risk_averse_prompts import (
    CLI_SYSTEM_PROMPT_SOURCE,
    DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE,
    MODEL_DEFAULT_NO_SYSTEM_PROMPT_SOURCE,
    default_system_prompt_for_dataset,
    model_uses_no_system_prompt,
    resolve_system_prompt,
)

EVAL_DIR = Path(__file__).resolve().parent
STEERING_DIR = EVAL_DIR.parent / "steering"
if str(STEERING_DIR) not in sys.path:
    sys.path.insert(0, str(STEERING_DIR))


class ResidualSteeringHook:
    """Simple residual stream steering hook used during generation."""

    def __init__(
        self,
        direction: torch.Tensor,
        alpha: float,
        apply_mode: str = "last_prompt_and_current",
        prompt_last_indices: Optional[List[int]] = None,
    ):
        self.direction = direction
        self.alpha = float(alpha)
        self.apply_mode = apply_mode
        self.prompt_last_indices = (
            None if prompt_last_indices is None else [int(index) for index in prompt_last_indices]
        )
        self._handle = None
        self._prefill_seen = False

    def _broadcast_direction(self, hidden: torch.Tensor) -> torch.Tensor:
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        while direction.dim() < hidden.dim():
            direction = direction.unsqueeze(0)
        return direction

    def _apply_all_positions(self, hidden: torch.Tensor) -> torch.Tensor:
        direction = self._broadcast_direction(hidden)
        return hidden + (self.alpha * direction)

    def _apply_last_prompt_and_current(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.dim() < 3 or hidden.shape[1] == 0:
            return self._apply_all_positions(hidden)

        steered = hidden.clone()
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)

        # The first hooked forward pass is the prompt prefill. Steer the final
        # non-padding prompt token for each example, then fall back to the final
        # sequence position on later decode steps. With use_cache=True, later
        # steps typically have sequence length 1, so this targets the current token.
        if not self._prefill_seen and hidden.shape[1] > 1 and self.prompt_last_indices is not None:
            batch_size = hidden.shape[0]
            if len(self.prompt_last_indices) != batch_size:
                raise ValueError(
                    f"prompt_last_indices batch mismatch: expected {batch_size}, got {len(self.prompt_last_indices)}"
                )
            batch_index = torch.arange(batch_size, device=hidden.device)
            token_index = torch.tensor(
                [max(0, min(index, hidden.shape[1] - 1)) for index in self.prompt_last_indices],
                device=hidden.device,
                dtype=torch.long,
            )
            steered[batch_index, token_index, :] = steered[batch_index, token_index, :] + (
                self.alpha * direction
            )
            self._prefill_seen = True
            return steered

        steered[:, -1, :] = steered[:, -1, :] + (self.alpha * direction)
        self._prefill_seen = True
        return steered

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            if not output:
                return output
            if self.apply_mode == "all_positions":
                steered_hidden = self._apply_all_positions(output[0])
            else:
                steered_hidden = self._apply_last_prompt_and_current(output[0])
            return (steered_hidden, *output[1:])
        if self.apply_mode == "all_positions":
            return self._apply_all_positions(output)
        return self._apply_last_prompt_and_current(output)

    def register(self, module):
        self._handle = module.register_forward_hook(self._hook)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# Flush output immediately so logs are visible in real time.
sys.stdout.reconfigure(line_buffering=True)
gc.collect()

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


def parse_alpha_list(value: str) -> List[float]:
    """Parse comma-separated alpha list."""
    alphas = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        alphas.append(float(raw))
    if not alphas:
        raise ValueError("No valid values parsed from --alphas")
    return alphas


def alpha_to_suffix(alpha: float) -> str:
    """Stable filename-safe suffix for alpha values."""
    prefix = "neg" if alpha < 0 else "pos"
    magnitude = f"{abs(alpha):g}".replace(".", "p")
    return f"{prefix}{magnitude}"


def format_repro_command(args, output_path: str, *, resume: bool) -> str:
    """Build a copy/paste command that reproduces current run settings."""
    cmd = ["python evaluate.py"]
    if args.model_path:
        cmd.extend(["--model_path", shlex.quote(str(args.model_path))])
    cmd.extend(["--base_model", shlex.quote(str(args.base_model))])

    if args.dataset == "custom":
        cmd.extend(["--custom_csv", shlex.quote(str(args.custom_csv))])
    else:
        cmd.extend(["--dataset", shlex.quote(str(args.dataset))])

    cmd.extend(["--num_situations", str(args.num_situations)])
    cmd.extend(["--start_position", str(args.start_position)])
    if args.end_position is not None:
        cmd.extend(["--end_position", str(args.end_position)])
    if args.stop_after is not None:
        cmd.extend(["--stop_after", str(args.stop_after)])
    cmd.extend(["--backend", shlex.quote(str(args.backend))])
    cmd.extend(["--temperature", str(args.temperature)])
    cmd.extend(["--top_p", str(args.top_p)])
    cmd.extend(["--top_k", str(args.top_k)])
    cmd.extend(["--seed", str(args.seed)])
    cmd.extend(["--max_new_tokens", str(args.max_new_tokens)])
    cmd.extend(["--max_time_per_generation", str(args.max_time_per_generation)])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--reasoning_max_tokens", str(args.reasoning_max_tokens)])

    if args.prompt_suffix:
        cmd.extend(["--prompt_suffix", shlex.quote(str(args.prompt_suffix))])
    if args.system_prompt:
        cmd.extend(["--system_prompt", shlex.quote(str(args.system_prompt))])
    if args.lin_only:
        cmd.append("--lin_only")
    if args.steering_direction_path:
        cmd.extend(["--steering_direction_path", shlex.quote(str(args.steering_direction_path))])
    if args.eval_layer is not None:
        cmd.extend(["--eval_layer", str(args.eval_layer)])
    if args.alphas:
        cmd.extend(["--alphas", shlex.quote(str(args.alphas))])
    if args.steering_apply_mode != "last_prompt_and_current":
        cmd.extend(["--steering_apply_mode", shlex.quote(str(args.steering_apply_mode))])
    if args.no_save_responses:
        cmd.append("--no_save_responses")
    if args.disable_thinking:
        cmd.append("--disable_thinking")
    if args.vllm_enable_prefix_caching:
        cmd.append("--vllm_enable_prefix_caching")
    else:
        cmd.append("--no-vllm_enable_prefix_caching")
    if resume:
        cmd.append("--resume")

    cmd.extend(["--output", shlex.quote(str(output_path))])
    return " ".join(cmd)


def print_stop_resume_banner(
    args,
    output_path: str,
    *,
    target_total: int,
    completed: int,
    pending_this_invocation: int,
):
    """Print a high-visibility stop/resume guide for explicit chunked runs."""
    if args.stop_after is None:
        return
    print("\n" + "!" * 88)
    print("IMPORTANT: CHUNKED EVAL MODE (STOP/RESUME QUICKSTART)")
    print("!" * 88)
    print(
        f"Target slice: {target_total} situations | already completed: {completed} | "
        f"planned this invocation: {pending_this_invocation}"
    )
    print(
        "Keep these fixed across chunks: --num_situations, --start_position, --end_position, "
        "and --output."
    )
    print(
        f"Current settings: --num_situations {args.num_situations}, --stop_after {args.stop_after}, "
        f"--start_position {args.start_position}, --end_position {args.end_position}"
    )

    first_chunk_cmd = format_repro_command(args, output_path, resume=False)
    resume_cmd = format_repro_command(args, output_path, resume=True)
    print("\nCopy/paste commands:")
    print(f"  First chunk:  {first_chunk_cmd}")
    print(f"  Resume next:  {resume_cmd}")

    if args.stop_after is not None:
        print(
            f"\nTo run this entire slice in one invocation, set --stop_after >= {target_total} "
            "(or set it exactly to your full target count)."
        )

    print("\nPerformance tip if generation is slow:")
    print("  1) Increase --batch_size and use --backend vllm")
    print("!" * 88 + "\n")


def get_input_device(model):
    """Best-effort model input device for tokenized tensors."""
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def get_decoder_layers(model):
    """Return decoder block list for common causal LM architectures.

    Handles standard (`.model.layers`), GPT-style (`.transformer.h`), and
    multimodal wrappers like Gemma-3, whose decoder blocks live under a nested
    text submodule (`.model.language_model.layers` in transformers,
    `.language_model.model.layers` in vLLM)."""
    candidates = (
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.model.language_model.layers,   # Gemma-3 multimodal (transformers)
        lambda m: m.language_model.model.layers,   # Gemma-3 multimodal (vLLM)
        lambda m: m.language_model.layers,
    )
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers
    raise ValueError(
        f"Unsupported model architecture for steering hooks: {type(model).__name__}"
    )


def load_steering_direction(path: str) -> torch.Tensor:
    """Load a steering vector from disk (tensor or dict wrapper)."""
    return load_steering_artifact(path)[0]


def load_steering_artifact(path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load a steering vector plus any saved metadata from disk."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(torch.float32).cpu(), {}
    if isinstance(obj, dict):
        metadata: dict[str, Any] = {}
        steering_info = obj.get("steering_info")
        if isinstance(steering_info, dict):
            metadata.update(steering_info)
        if "layer" in obj and "layer" not in metadata:
            metadata["layer"] = obj["layer"]
        for key in ("direction", "vector", "steering_direction"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value.detach().to(torch.float32).cpu(), metadata
            if isinstance(value, (list, tuple)):
                return torch.tensor(value, dtype=torch.float32), metadata
    if isinstance(obj, (list, tuple)):
        return torch.tensor(obj, dtype=torch.float32), {}
    raise ValueError(
        f"Unsupported steering direction payload at {path}. "
        "Expected Tensor, list, or dict with a direction tensor/list."
    )


def convert_numpy(obj):
    """Convert numpy/torch scalar-like values to native Python for JSON."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


def apply_chat_template_safe(tokenizer, messages, disable_thinking: bool) -> str:
    """Apply chat template, tolerating tokenizers without enable_thinking support."""
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=not disable_thinking,
            **template_kwargs,
        )
    except TypeError:
        if disable_thinking:
            try:
                return tokenizer.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
            except TypeError:
                pass
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def build_messages(eval_prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    """Build chat messages for one evaluation request."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": eval_prompt})
    return messages


def count_generated_tokens(
    output_ids: torch.Tensor,
    *,
    prompt_token_count: int,
    prompt_length: int,
    pad_token_id: Optional[int],
) -> int:
    """Count generated tokens for one row in a padded batch."""
    if pad_token_id is None:
        return int(max(output_ids.shape[0] - prompt_token_count, 0))
    total_non_pad_tokens = int(output_ids.ne(pad_token_id).sum().item())
    return max(total_non_pad_tokens - int(prompt_length), 0)


def vllm_settings_from_args(args) -> Dict[str, Any]:
    """Serialize vLLM runtime settings into the output JSON."""
    return {
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len,
        "dtype": args.vllm_dtype,
        "enable_prefix_caching": args.vllm_enable_prefix_caching,
        "max_lora_rank": args.vllm_max_lora_rank if args.model_path else None,
    }


def load_vllm_engine(args):
    """Lazily construct a vLLM engine and optional LoRA request."""
    # Keep vLLM imports on the PyTorch path; TensorFlow imports were a source of
    # environment breakage on Lambda images when transformers was imported transitively.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_TORCH", "1")

    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise ImportError(
            "vLLM backend requested, but `vllm` is not installed. "
            "Install it on the GPU host, then re-run with --backend vllm."
        ) from exc

    llm_kwargs: Dict[str, Any] = {
        "model": args.base_model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "dtype": args.vllm_dtype,
    }
    if args.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = args.vllm_max_model_len
    llm_kwargs["enable_prefix_caching"] = args.vllm_enable_prefix_caching
    if getattr(args, "_vllm_enforce_eager", False):
        # Steering registers a forward hook inside the vLLM worker (via apply_model).
        # Eager mode keeps the hook live (CUDA-graph capture would bypass it), and
        # apply_model needs insecure (pickle) serialization to ship our function to
        # the worker. See vllm_steering.py.
        llm_kwargs["enforce_eager"] = True
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        # Prefix caching keys on token ids, not activations, so it would reuse KV
        # across different alphas/directions and silently drop the prompt-side
        # steering (the first alpha's prompt KV gets reused by all later alphas).
        # It MUST be off for activation steering.
        llm_kwargs["enable_prefix_caching"] = False
    if args.model_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.vllm_max_lora_rank

    engine = LLM(**llm_kwargs)

    lora_request = None
    if args.model_path:
        adapter_name = Path(args.model_path).resolve().name
        lora_request = LoRARequest(adapter_name, 1, str(Path(args.model_path).resolve()))

    return engine, lora_request


def summarize_results(results):
    """Compute aggregate metrics from per-situation result records."""
    valid = [r for r in results if r["option_type"] is not None]
    behavioral_valid = [r for r in valid if r["option_type"] in BEHAVIORAL_OPTION_TYPES]
    if valid:
        if behavioral_valid:
            cooperate_rate = sum(r["option_type"] == "Cooperate" for r in behavioral_valid) / len(behavioral_valid)
            rebel_rate = sum(r["option_type"] == "Rebel" for r in behavioral_valid) / len(behavioral_valid)
            steal_rate = sum(r["option_type"] == "Steal" for r in behavioral_valid) / len(behavioral_valid)
        else:
            cooperate_rate = rebel_rate = steal_rate = None
        cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid)
        linear_valid = [r for r in valid if r.get("is_best_linear") is not None]
        linear_rate = sum(r["is_best_linear"] for r in linear_valid) / len(linear_valid) if linear_valid else 0
        worst_linear_valid = [r for r in valid if r.get("is_worst_linear") is not None]
        worst_linear_rate = (
            sum(r["is_worst_linear"] for r in worst_linear_valid) / len(worst_linear_valid)
            if worst_linear_valid
            else 0
        )
        ev_ratio_valid = [r for r in valid if r.get("expected_value_fraction_of_best") is not None]
        ev_relative_valid = [r for r in valid if r.get("expected_value_relative_to_range") is not None]
        ev_regret_valid = [r for r in valid if r.get("expected_value_regret") is not None]
        avg_ev_fraction_of_best = (
            sum(float(r["expected_value_fraction_of_best"]) for r in ev_ratio_valid) / len(ev_ratio_valid)
            if ev_ratio_valid
            else None
        )
        avg_ev_relative_to_range = (
            sum(float(r["expected_value_relative_to_range"]) for r in ev_relative_valid) / len(ev_relative_valid)
            if ev_relative_valid
            else None
        )
        avg_ev_regret = (
            sum(float(r["expected_value_regret"]) for r in ev_regret_valid) / len(ev_regret_valid)
            if ev_regret_valid
            else None
        )
    else:
        cooperate_rate = rebel_rate = steal_rate = cara_rate = linear_rate = 0
        worst_linear_rate = 0
        avg_ev_fraction_of_best = None
        avg_ev_relative_to_range = None
        avg_ev_regret = None

    parse_rate = len(valid) / len(results) if results else 0
    return {
        "parse_rate": parse_rate,
        "cooperate_rate": cooperate_rate,
        "rebel_rate": rebel_rate,
        "steal_rate": steal_rate,
        "best_cara_rate": cara_rate,
        "best_linear_rate": linear_rate,
        "worst_linear_rate": worst_linear_rate,
        "avg_expected_value_fraction_of_best": avg_ev_fraction_of_best,
        "avg_expected_value_relative_to_range": avg_ev_relative_to_range,
        "avg_expected_value_regret": avg_ev_regret,
    }


def summarize_result_payload(results: List[Dict]) -> Dict:
    """Return metrics plus counts using the existing rate semantics."""
    valid = [r for r in results if r["option_type"] is not None]
    behavioral_valid = [r for r in valid if r["option_type"] in BEHAVIORAL_OPTION_TYPES]
    return {
        "metrics": summarize_results(results),
        "num_valid": len(valid),
        "num_behaviorally_classified": len(behavioral_valid),
        "num_total": len(results),
        "num_parse_failed": count_parse_failures(results),
    }


def summarize_manifest_counts(
    situation_manifest: List[Dict],
    *,
    field_name: str,
    ordered_values: List[str],
) -> Dict:
    """Count selected situations by one ordered manifest field."""
    counts = {}
    for value in ordered_values:
        count = sum(1 for entry in situation_manifest if entry.get(field_name) == value)
        if count:
            counts[value] = count
    return counts


def summarize_results_by_field(
    results: List[Dict],
    situation_manifest: List[Dict],
    *,
    field_name: str,
    ordered_values: List[str],
) -> Dict:
    """Compute the standard metric bundle for one ordered manifest field."""
    target_ids_by_value = {value: [] for value in ordered_values}
    for entry in situation_manifest:
        value = entry.get(field_name)
        if value in target_ids_by_value:
            target_ids_by_value[value].append(entry["situation_id"])

    summarized = {}
    for value in ordered_values:
        target_ids = target_ids_by_value[value]
        if not target_ids:
            continue
        field_results = [row for row in results if row.get(field_name) == value]
        summarized[value] = summarize_result_payload(field_results)
    return summarized


def summarize_progress_by_field(
    results: List[Dict],
    situation_manifest: List[Dict],
    *,
    field_name: str,
    ordered_values: List[str],
) -> Dict:
    """Track completion progress separately for one ordered manifest field."""
    completed_ids = {row.get("situation_id") for row in results if row.get("situation_id") is not None}
    progress = {}
    for value in ordered_values:
        field_ids = [entry["situation_id"] for entry in situation_manifest if entry.get(field_name) == value]
        if not field_ids:
            continue
        completed = sum(1 for sid in field_ids if sid in completed_ids)
        next_situation_id = next((sid for sid in field_ids if sid not in completed_ids), None)
        progress[value] = {
            "target_total": len(field_ids),
            "completed": completed,
            "remaining": max(len(field_ids) - completed, 0),
            "next_situation_id": next_situation_id,
        }
    return progress


def project_result_row_for_output(row: Dict, *, include_response: bool) -> Dict:
    """Persist only the per-situation fields intended for analysis."""
    keys = [
        "situation_id",
        "dataset_position",
        "subset_type",
        "source_stakes",
        "source_condition",
        "source_csv_name",
        "source_situation_id",
        "option_types_besides_cooperate",
        "prompt",
        "num_options",
        "probability_format",
        "choice",
        "choice_index",
        "parser_strategy",
        "num_tokens_generated",
        "generation_batch_time_seconds",
        "generation_batch_size",
        "generation_finish_reason",
        "option_type",
        "is_best_cara",
        "is_best_linear",
        "is_worst_linear",
        "expected_value",
        "max_expected_value",
        "min_expected_value",
        "expected_value_fraction_of_best",
        "expected_value_relative_to_range",
        "expected_value_regret",
    ]
    projected = {key: row.get(key) for key in keys}
    stop_reason = row.get("generation_stop_reason")
    finish_reason = row.get("generation_finish_reason")
    if stop_reason and stop_reason != finish_reason:
        projected["generation_stop_reason"] = stop_reason
    if include_response:
        projected["response"] = row.get("response")
    return projected


def project_failed_response_for_output(row: Dict) -> Dict:
    """Persist a compact sample of parse failures."""
    keys = [
        "situation_id",
        "dataset_position",
        "subset_type",
        "source_stakes",
        "source_condition",
        "source_csv_name",
        "source_situation_id",
        "option_types_besides_cooperate",
        "num_options",
        "prompt",
        "parser_strategy",
        "response",
    ]
    return {key: row.get(key) for key in keys}


def format_pct_metric(value: Optional[float]) -> str:
    """Format percentage-like metrics, allowing None for n/a slices."""
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


def count_parse_failures(results: List[Dict]) -> int:
    """Count situations where parser failed to extract a valid option."""
    return sum(1 for row in results if row.get("option_type") is None)


def atomic_write_json(path: str, payload: Dict):
    """Write JSON atomically to reduce corruption risk on interruption."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, output_path)


def compact_results_for_resume(results: List[Dict]) -> List[Dict]:
    """Persist only fields needed for resume + metric recomputation."""
    return [project_result_row_for_output(row, include_response=False) for row in results]


def drop_response_text(results: List[Dict]) -> List[Dict]:
    """Drop full response text while keeping prompts and metrics fields."""
    return [project_result_row_for_output(row, include_response=False) for row in results]


def dedupe_results_by_situation(results: List[Dict], ordered_situation_ids: List[int]) -> List[Dict]:
    """Deduplicate by situation_id and preserve dataset order."""
    latest_by_id = {}
    for row in results:
        sid = row.get("situation_id")
        if sid is None:
            continue
        latest_by_id[sid] = row

    deduped = [latest_by_id[sid] for sid in ordered_situation_ids if sid in latest_by_id]
    return deduped


def load_existing_run_state(
    output_path: str,
    ordered_situation_ids: List[int],
    *,
    allow_backup_fallback: bool = True,
):
    """Load resumable state from output JSON (or .bak fallback)."""
    candidates = [Path(output_path)]
    if allow_backup_fallback:
        candidates.append(Path(f"{output_path}.bak"))

    loaded = None
    loaded_from = None
    last_error = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r") as f:
                loaded = json.load(f)
            loaded_from = str(candidate)
            break
        except Exception as exc:
            last_error = exc

    if loaded is None:
        if last_error is not None:
            raise RuntimeError(
                f"Found prior output but failed to parse JSON: {output_path} ({last_error})"
            ) from last_error
        return None

    results = loaded.get("results")
    if not isinstance(results, list):
        results = loaded.get("resume_records")
    if not isinstance(results, list):
        raise ValueError(
            "Cannot resume: output JSON does not contain resumable records. "
            "Expected `results` or `resume_records` as a list."
        )

    ordered_id_set = set(ordered_situation_ids)
    rows_in_target = [r for r in results if r.get("situation_id") in ordered_id_set]
    deduped_results = dedupe_results_by_situation(results, ordered_situation_ids)
    dropped_duplicates = max(len(rows_in_target) - len(deduped_results), 0)

    failed = loaded.get("failed_responses_sample")
    if not isinstance(failed, list):
        failed = loaded.get("failed_responses")
    if not isinstance(failed, list):
        failed = []

    return {
        "loaded_from": loaded_from,
        "payload": loaded,
        "results": deduped_results,
        "failed_responses": failed,
        "dropped_duplicates": dropped_duplicates,
    }


def save_incremental(
    output_path,
    args,
    results,
    failed_responses,
    situations_evaluated,
    target_situations,
    *,
    steering_alpha: float,
    steering_info: Optional[Dict] = None,
    create_backup: bool = False,
):
    """Save current run state to disk for crash resilience."""
    situation_manifest = build_situation_manifest(target_situations)
    situation_index = {entry["situation_id"]: entry for entry in situation_manifest}
    annotate_rows_with_situation_metadata(results, situation_index)
    annotate_rows_with_situation_metadata(failed_responses, situation_index)
    summary_payload = summarize_result_payload(results)
    metrics = summary_payload["metrics"]
    done_ids = {r.get("situation_id") for r in results if r.get("situation_id") is not None}
    target_situation_ids = [entry["situation_id"] for entry in situation_manifest]
    target_total = len(target_situation_ids)
    target_completed = sum(1 for sid in target_situation_ids if sid in done_ids)
    next_situation_id = next((sid for sid in target_situation_ids if sid not in done_ids), None)
    selected_subset_type_counts = summarize_manifest_counts(
        situation_manifest,
        field_name="subset_type",
        ordered_values=list(SUBSET_TYPES),
    )
    selected_probability_format_counts = summarize_manifest_counts(
        situation_manifest,
        field_name="probability_format",
        ordered_values=list(PROBABILITY_FORMATS),
    )

    eval_cfg = {
        "backend": args.backend,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "reasoning": {"max_tokens": args.reasoning_max_tokens},
        "enable_thinking": not args.disable_thinking,
        "num_situations": target_total,
        "num_situations_completed": target_completed,
        "start_position": args.start_position,
        "end_position": args.end_position,
        "stop_after": args.stop_after,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "dataset_base_alias": args.dataset_base_alias,
        "dataset_variant": args.resolved_dataset_variant,
        "custom_csv": args.custom_csv,
        "csv_path": args.csv_path,
        "lin_only": args.lin_only,
        "batch_size": args.batch_size,
        "system_prompt": args.system_prompt,
        "system_prompt_source": getattr(args, "system_prompt_source", None),
        "prompt_suffix": args.prompt_suffix,
        "steering_alpha": steering_alpha,
        "steering_apply_mode": args.steering_apply_mode,
        "selected_situation_ids": target_situation_ids,
        "selected_subset_type_counts": selected_subset_type_counts,
        "selected_probability_format_counts": selected_probability_format_counts,
        "selected_situations": situation_manifest,
    }
    if args.backend == "vllm":
        eval_cfg["vllm"] = vllm_settings_from_args(args)
    if steering_info:
        eval_cfg["steering"] = steering_info

    parse_failed_total = summary_payload["num_parse_failed"]
    failed_sample = [project_failed_response_for_output(row) for row in failed_responses[-10:]]
    stored_results = results if not args.no_save_responses else drop_response_text(results)
    if not args.no_save_responses:
        stored_results = [project_result_row_for_output(row, include_response=True) for row in results]
    subset_metrics = summarize_results_by_field(
        results,
        situation_manifest,
        field_name="subset_type",
        ordered_values=list(SUBSET_TYPES),
    )
    probability_format_metrics = summarize_results_by_field(
        results,
        situation_manifest,
        field_name="probability_format",
        ordered_values=list(PROBABILITY_FORMATS),
    )
    source_stakes_metrics = summarize_results_by_field(
        results,
        situation_manifest,
        field_name="source_stakes",
        ordered_values=list(SOURCE_STAKES),
    )
    metrics_by_subset_type = {**subset_metrics, **probability_format_metrics}
    progress_by_subset_type = {
        **summarize_progress_by_field(
            results,
            situation_manifest,
            field_name="subset_type",
            ordered_values=list(SUBSET_TYPES),
        ),
        **summarize_progress_by_field(
            results,
            situation_manifest,
            field_name="probability_format",
            ordered_values=list(PROBABILITY_FORMATS),
        ),
    }
    progress_by_source_stakes = summarize_progress_by_field(
        results,
        situation_manifest,
        field_name="source_stakes",
        ordered_values=list(SOURCE_STAKES),
    )

    output_data = convert_numpy(
        {
            "evaluation_config": eval_cfg,
            "metrics": metrics,
            "num_valid": summary_payload["num_valid"],
            "num_behaviorally_classified": summary_payload["num_behaviorally_classified"],
            "num_total": summary_payload["num_total"],
            "num_parse_failed": parse_failed_total,
            "metrics_by_subset_type": metrics_by_subset_type,
            "metrics_by_probability_format": probability_format_metrics,
            "metrics_by_source_stakes": source_stakes_metrics,
            "results": stored_results,
            "resume_records": compact_results_for_resume(results),
            "failed_responses": failed_sample,  # Backwards-compatible key name.
            "failed_responses_sample": failed_sample,
            "progress": {
                "target_total": target_total,
                "completed": target_completed,
                "remaining": max(target_total - target_completed, 0),
                "next_situation_id": next_situation_id,
                "checkpoint_index": situations_evaluated,
            },
            "progress_by_subset_type": progress_by_subset_type,
            "progress_by_probability_format": summarize_progress_by_field(
                results,
                situation_manifest,
                field_name="probability_format",
                ordered_values=list(PROBABILITY_FORMATS),
            ),
            "progress_by_source_stakes": progress_by_source_stakes,
        }
    )

    atomic_write_json(output_path, output_data)
    if create_backup:
        backup_path = f"{output_path}.bak"
        shutil.copy2(output_path, backup_path)


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


def generate_response_transformers(
    *,
    model,
    tokenizer,
    eval_prompts: List[str],
    system_prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    max_time_per_generation: float,
    disable_thinking: bool,
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
    steering_alpha: float = 0.0,
    steering_apply_mode: str = "last_prompt_and_current",
):
    """Generate one or more responses with the Transformers backend."""
    texts = [
        apply_chat_template_safe(
            tokenizer,
            build_messages(eval_prompt, system_prompt),
            disable_thinking=disable_thinking,
        )
        for eval_prompt in eval_prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(get_input_device(model))
    prompt_token_count = inputs["input_ids"].shape[1]
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    prompt_last_indices = (
        (inputs["attention_mask"].to(torch.long) * torch.arange(prompt_token_count, device=inputs["attention_mask"].device))
        .max(dim=1)
        .values
        .tolist()
    )

    hook = None
    if steering_block is not None and steering_direction is not None and abs(steering_alpha) > 0:
        block_device = next(steering_block.parameters()).device
        direction = steering_direction.to(device=block_device, dtype=model.dtype)
        hook = ResidualSteeringHook(
            direction=direction,
            alpha=steering_alpha,
            apply_mode=steering_apply_mode,
            prompt_last_indices=prompt_last_indices,
        ).register(steering_block)

    gen_start = time.time()
    try:
        with torch.inference_mode():
            if temperature == 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation,
                )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and len(eval_prompts) > 1:
            print(
                f"  WARNING: CUDA OOM while generating batch of {len(eval_prompts)}. "
                "Retrying sequentially."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            responses = []
            generated_token_counts = []
            total_elapsed = 0.0
            metadata = []
            for eval_prompt in eval_prompts:
                sub_responses, sub_token_counts, sub_elapsed, sub_metadata = generate_response_transformers(
                    model=model,
                    tokenizer=tokenizer,
                    eval_prompts=[eval_prompt],
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    max_time_per_generation=max_time_per_generation,
                    disable_thinking=disable_thinking,
                    steering_block=steering_block,
                    steering_direction=steering_direction,
                    steering_alpha=steering_alpha,
                    steering_apply_mode=steering_apply_mode,
                )
                responses.extend(sub_responses)
                generated_token_counts.extend(sub_token_counts)
                total_elapsed += sub_elapsed
                metadata.extend(sub_metadata)
            return responses, generated_token_counts, total_elapsed, metadata
        raise
    finally:
        if hook is not None:
            hook.remove()

    gen_elapsed = time.time() - gen_start
    responses = []
    generated_token_counts = []
    metadata = []
    for row_idx, output_ids in enumerate(outputs):
        gen_ids = output_ids[prompt_token_count:]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        token_count = count_generated_tokens(
            output_ids,
            prompt_token_count=prompt_token_count,
            prompt_length=int(prompt_lengths[row_idx]),
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_token_counts.append(token_count)
        metadata.append(
            {
                "finish_reason": "length" if token_count >= max_new_tokens else "eos_or_stop",
                "stop_reason": None,
            }
        )
    return responses, generated_token_counts, gen_elapsed, metadata


def generate_response_vllm(
    *,
    model,
    eval_prompts: List[str],
    system_prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    max_new_tokens: int,
    disable_thinking: bool,
    lora_request=None,
):
    """Generate one or more responses with the vLLM backend."""
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM backend requested, but `vllm` is not installed. "
            "Install it on the GPU host, then re-run with --backend vllm."
        ) from exc

    batch_messages = [
        build_messages(eval_prompt, system_prompt)
        for eval_prompt in eval_prompts
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        max_tokens=max_new_tokens,
        ignore_eos=False,
    )

    call_kwargs: Dict[str, Any] = {
        "messages": batch_messages,
        "sampling_params": sampling_params,
        "use_tqdm": False,
        "chat_template_kwargs": {"enable_thinking": not disable_thinking},
    }
    if lora_request is not None:
        call_kwargs["lora_request"] = lora_request

    gen_start = time.time()
    try:
        outputs = model.chat(**call_kwargs)
    except TypeError:
        call_kwargs.pop("chat_template_kwargs", None)
        outputs = model.chat(**call_kwargs)
    gen_elapsed = time.time() - gen_start

    responses = []
    generated_token_counts = []
    metadata = []
    for request_output in outputs:
        if not getattr(request_output, "outputs", None):
            responses.append("")
            generated_token_counts.append(0)
            metadata.append({"finish_reason": None, "stop_reason": None})
            continue
        completion = request_output.outputs[0]
        responses.append(completion.text)
        token_ids = getattr(completion, "token_ids", None) or []
        generated_token_counts.append(len(token_ids))
        metadata.append(
            {
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
            }
        )
    return responses, generated_token_counts, gen_elapsed, metadata


def generate_response(
    *,
    backend: str,
    model,
    tokenizer,
    eval_prompts: List[str],
    system_prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    max_new_tokens: int,
    max_time_per_generation: float,
    disable_thinking: bool,
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
    steering_alpha: float = 0.0,
    steering_apply_mode: str = "last_prompt_and_current",
    lora_request=None,
):
    """Dispatch generation to the selected inference backend."""
    if backend == "vllm":
        return generate_response_vllm(
            model=model,
            eval_prompts=eval_prompts,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_new_tokens=max_new_tokens,
            disable_thinking=disable_thinking,
            lora_request=lora_request,
        )
    return generate_response_transformers(
        model=model,
        tokenizer=tokenizer,
        eval_prompts=eval_prompts,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        max_time_per_generation=max_time_per_generation,
        disable_thinking=disable_thinking,
        steering_block=steering_block,
        steering_direction=steering_direction,
        steering_alpha=steering_alpha,
        steering_apply_mode=steering_apply_mode,
    )


def run_single_alpha_eval(
    *,
    backend: str,
    model,
    tokenizer,
    situations,
    args,
    output_path: str,
    steering_alpha: float,
    steering_info: Optional[Dict],
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
    lora_request=None,
):
    """Run one evaluation pass for a single alpha value."""
    if backend == "vllm":
        # vLLM steering: the hook is registered once on the engine; set its alpha
        # for this pass (no-op / clean baseline when alpha == 0 or no controller).
        _vllm_ctrl = getattr(model, "_steering_controller", None)
        if _vllm_ctrl is not None:
            _vllm_ctrl.set_alpha(steering_alpha)
    situation_manifest = build_situation_manifest(situations)
    situation_index = build_situation_manifest_index(situations)
    target_situation_ids = [sit["situation_id"] for sit in situations]
    print(f"Evaluating on {len(situations)} situations with PERMISSIVE parser...")
    print(f"Backend: {backend}")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'sampling'})")
    if abs(args.temperature - DEFAULT_EVAL_TEMPERATURE) > 1e-12:
        print(
            f"WARNING: Non-default temperature in use ({args.temperature}). "
            f"The canonical paper default is {DEFAULT_EVAL_TEMPERATURE}."
        )
    print(f"Steering alpha: {steering_alpha:+.4f}")
    print(f"Steering apply mode: {args.steering_apply_mode}")
    print(f"Top-p: {args.top_p}")
    print(f"Top-k: {args.top_k}")
    print(f"Seed: {args.seed}")
    print(f"Dataset variant: {args.resolved_dataset_variant}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max time per generation: {args.max_time_per_generation}s")
    print(f"Thinking mode: {'DISABLED' if args.disable_thinking else 'ENABLED'}")
    system_prompt_source = getattr(args, "system_prompt_source", "unknown")
    if args.system_prompt:
        print(f"System prompt: YES ({len(args.system_prompt)} chars; source: {system_prompt_source})")
    else:
        print(f"System prompt: NO (source: {system_prompt_source})")
    if backend == "vllm":
        print(
            "vLLM settings: "
            f"tp={args.vllm_tensor_parallel_size}, "
            f"gpu_mem={args.vllm_gpu_memory_utilization}, "
            f"prefix_cache={'ON' if args.vllm_enable_prefix_caching else 'OFF'}"
        )
        print("Note: vLLM backend does not enforce --max_time_per_generation per batch.")
    if args.no_save_responses:
        print(
            "Saving responses: NO (--no_save_responses, strongly discouraged for paper replication)"
        )
    else:
        print("Saving responses: YES (default and strongly recommended)")
    print(f"Checkpoint frequency: every {args.save_every} situation(s)")
    if args.backup_every > 0:
        print(f"Backup frequency: every {args.backup_every} situation(s) -> {output_path}.bak")
    if args.save_every % args.batch_size != 0:
        print(
            f"Note: --save_every {args.save_every} is not a multiple of --batch_size {args.batch_size}; "
            "checkpoints happen at batch boundaries, so the effective save cadence may differ."
        )
    if args.backup_every > 0 and args.backup_every % args.batch_size != 0:
        print(
            f"Note: --backup_every {args.backup_every} is not a multiple of --batch_size {args.batch_size}; "
            "backups happen at batch boundaries, so the effective backup cadence may differ."
        )
    print(f"Results will be saved incrementally to: {output_path}")
    print()

    results = []
    failed_responses = []
    generation_times = []
    completed_ids = set()
    resumed_count = 0

    if args.resume:
        prior_state = load_existing_run_state(output_path, target_situation_ids)
        if prior_state is not None:
            results = prior_state["results"]
            failed_responses = prior_state["failed_responses"]
            annotate_rows_with_situation_metadata(results, situation_index)
            annotate_rows_with_situation_metadata(failed_responses, situation_index)
            completed_ids = {r.get("situation_id") for r in results if r.get("situation_id") is not None}
            resumed_count = len(completed_ids)
            loaded_from = prior_state["loaded_from"]
            print(f"Resuming from existing checkpoint: {loaded_from}")
            print(f"Already completed: {resumed_count}/{len(situations)} situations")
            dropped_duplicates = int(prior_state.get("dropped_duplicates", 0) or 0)
            if dropped_duplicates > 0:
                print(
                    f"WARNING: Dropped {dropped_duplicates} duplicate checkpoint rows by situation_id "
                    "while resuming."
                )

            prior_cfg = prior_state["payload"].get("evaluation_config", {})
            prior_dataset = prior_cfg.get("dataset")
            prior_csv_path = (
                prior_cfg.get("csv_path")
                or prior_cfg.get("custom_csv")
                or prior_cfg.get("val_csv")
            )
            if prior_dataset and prior_dataset != args.dataset:
                print(
                    f"WARNING: Resume dataset mismatch (checkpoint={prior_dataset}, current={args.dataset}). "
                    "Proceeding with current target slice."
                )
            if prior_csv_path and str(prior_csv_path) != str(args.csv_path):
                print(
                    "WARNING: Resume CSV path differs from current run.\n"
                    f"  checkpoint: {prior_csv_path}\n"
                    f"  current:    {args.csv_path}"
                )
            for field in (
                "backend",
                "base_model",
                "model_path",
                "temperature",
                "top_p",
                "top_k",
                "seed",
                "max_new_tokens",
                "start_position",
                "end_position",
            ):
                prior_value = prior_cfg.get(field)
                current_value = getattr(args, field, None)
                if prior_value is not None and prior_value != current_value:
                    print(
                        f"WARNING: Resume {field} differs from checkpoint "
                        f"(checkpoint={prior_value}, current={current_value})."
                    )
        else:
            print("Resume requested but no prior checkpoint found; starting fresh.")
    elif Path(output_path).exists():
        raise FileExistsError(
            "Output file already exists. To continue the interrupted run, re-run with "
            f"--resume --output {output_path}. To start fresh, choose a new --output path "
            "or delete the old output file first."
        )

    pending_situations = [sit for sit in situations if sit["situation_id"] not in completed_ids]
    if args.stop_after is not None:
        pending_situations = pending_situations[: args.stop_after]
        print(f"Stop-after mode: evaluating at most {len(pending_situations)} new situations this run.")

    print_stop_resume_banner(
        args=args,
        output_path=output_path,
        target_total=len(situations),
        completed=len(completed_ids),
        pending_this_invocation=len(pending_situations),
    )

    if not pending_situations:
        print("No pending situations for this run. Writing fresh summary from existing checkpoint data.")
        save_incremental(
            output_path,
            args,
            results,
            failed_responses,
            len(results),
            situations,
            steering_alpha=steering_alpha,
            steering_info=steering_info,
            create_backup=True,
        )
        summary_payload = summarize_result_payload(results)
        metrics = summary_payload["metrics"]
        return {
            "output_path": output_path,
            "alpha": steering_alpha,
            "metrics": metrics,
            "num_valid": summary_payload["num_valid"],
            "num_total": summary_payload["num_total"],
            "num_parse_failed": summary_payload["num_parse_failed"],
            "num_resumed": resumed_count,
            "num_new": 0,
        }

    for sit in pending_situations:
        sit["eval_prompt"] = build_eval_prompt(sit["prompt_raw"], args.prompt_suffix)
    print(f"Prepared prompts for {len(pending_situations)} situation(s).")

    eval_start_time = time.time()
    session_evaluated = 0
    for batch_start in range(0, len(pending_situations), args.batch_size):
        batch = pending_situations[batch_start : batch_start + args.batch_size]
        batch_prompts = [sit["eval_prompt"] for sit in batch]
        prior_session_evaluated = session_evaluated

        responses, generated_token_counts, batch_elapsed, generation_metadata = generate_response(
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            eval_prompts=batch_prompts,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            max_time_per_generation=args.max_time_per_generation,
            disable_thinking=args.disable_thinking,
            steering_block=steering_block,
            steering_direction=steering_direction,
            steering_alpha=steering_alpha,
            steering_apply_mode=args.steering_apply_mode,
            lora_request=lora_request,
        )
        effective_elapsed = batch_elapsed / max(1, len(batch))

        for batch_offset, (sit, eval_prompt, response, num_generated_tokens, metadata) in enumerate(
            zip(batch, batch_prompts, responses, generated_token_counts, generation_metadata)
        ):
            parse_result = parse_choice_with_strategy(
                response,
                sit["num_options"],
                label_style=sit.get("answer_label_style"),
                finish_reason=metadata.get("finish_reason"),
            )
            choice = parse_result.choice
            parser_strategy = parse_result.strategy
            choice_index = label_to_option_number(choice) if choice else None

            result_row = {
                "situation_id": sit["situation_id"],
                "dataset_position": sit["dataset_position"],
                "subset_type": sit["subset_type"],
                "source_stakes": sit.get("source_stakes"),
                "source_condition": sit.get("source_condition"),
                "source_csv_name": sit.get("source_csv_name"),
                "source_situation_id": sit.get("source_situation_id"),
                "option_types_besides_cooperate": sit["option_types_besides_cooperate"],
                "prompt": eval_prompt,
                "num_options": sit["num_options"],
                "probability_format": sit["probability_format"],
                "bucket_label": sit["bucket_label"],
                "choice": choice if choice and choice in sit["options"] else None,
                "choice_index": choice_index if choice and choice in sit["options"] else None,
                "parser_strategy": parser_strategy,
                "response": None if args.no_save_responses else response,
                "response_length": len(response),
                "num_tokens_generated": int(num_generated_tokens),
                "generation_time_seconds": round(effective_elapsed, 2),
                "generation_batch_time_seconds": round(batch_elapsed, 2),
                "generation_batch_size": len(batch),
                "generation_finish_reason": metadata.get("finish_reason"),
                "generation_stop_reason": metadata.get("stop_reason"),
            }

            if choice and choice in sit["options"]:
                chosen = sit["options"][choice]
                result_row.update(
                    {
                        "option_type": chosen["type"],
                        "is_best_cara": chosen["is_best_cara"],
                        "is_best_linear": chosen["is_best_linear"],
                        "is_worst_linear": chosen.get("is_worst_linear"),
                        "expected_value": chosen.get("expected_value"),
                        "max_expected_value": sit.get("max_expected_value"),
                        "min_expected_value": sit.get("min_expected_value"),
                        "expected_value_fraction_of_best": (
                            (chosen.get("expected_value") / sit.get("max_expected_value"))
                            if chosen.get("expected_value") is not None
                            and sit.get("max_expected_value") not in (None, 0)
                            else None
                        ),
                        "expected_value_relative_to_range": (
                            1.0
                            if chosen.get("expected_value") is not None
                            and sit.get("max_expected_value") is not None
                            and sit.get("min_expected_value") is not None
                            and abs(sit.get("max_expected_value") - sit.get("min_expected_value")) < 1e-12
                            else (
                                (chosen.get("expected_value") - sit.get("min_expected_value"))
                                / (sit.get("max_expected_value") - sit.get("min_expected_value"))
                            )
                            if chosen.get("expected_value") is not None
                            and sit.get("max_expected_value") is not None
                            and sit.get("min_expected_value") is not None
                            else None
                        ),
                        "expected_value_regret": (
                            sit.get("max_expected_value") - chosen.get("expected_value")
                            if chosen.get("expected_value") is not None
                            and sit.get("max_expected_value") is not None
                            else None
                        ),
                    }
                )
            else:
                result_row.update(
                    {
                        "option_type": None,
                        "is_best_cara": None,
                        "is_best_linear": None,
                        "is_worst_linear": None,
                        "expected_value": None,
                        "max_expected_value": sit.get("max_expected_value"),
                        "min_expected_value": sit.get("min_expected_value"),
                        "expected_value_fraction_of_best": None,
                        "expected_value_relative_to_range": None,
                        "expected_value_regret": None,
                    }
                )
                failed_responses.append(
                    {
                        "situation_id": sit["situation_id"],
                        "dataset_position": sit["dataset_position"],
                        "subset_type": sit["subset_type"],
                        "source_stakes": sit.get("source_stakes"),
                        "source_condition": sit.get("source_condition"),
                        "source_csv_name": sit.get("source_csv_name"),
                        "source_situation_id": sit.get("source_situation_id"),
                        "option_types_besides_cooperate": sit["option_types_besides_cooperate"],
                        "num_options": sit["num_options"],
                        "prompt": eval_prompt,
                        "parser_strategy": parser_strategy,
                        "response": response,
                    }
                )
                if len(failed_responses) > 100:
                    failed_responses = failed_responses[-100:]

            results.append(result_row)
            completed_ids.add(sit["situation_id"])
            session_evaluated += 1
            generation_times.append(effective_elapsed)
            avg_time = sum(generation_times) / len(generation_times)
            remaining_situations = len(situations) - len(completed_ids)
            remaining = avg_time * remaining_situations

            status = "OK" if result_row["choice"] else "PARSE_FAIL"
            strategy_text = parser_strategy if parser_strategy else "none"
            timing_text = (
                f"{effective_elapsed:.1f}s/item ({batch_elapsed:.1f}s batch x{len(batch)})"
                if len(batch) > 1
                else f"{batch_elapsed:.1f}s"
            )
            print(
                f"  [{len(completed_ids)}/{len(situations)}] sit_id={sit['situation_id']} | {status} "
                f"({strategy_text}) | {int(num_generated_tokens)} tokens | {timing_text} | "
                f"ETA: {remaining/60:.1f}min"
            )

            if effective_elapsed > 60:
                print(
                    f"  WARNING: Effective per-example generation time was {effective_elapsed:.0f}s (>60s). "
                    "Model may be generating excessively long output."
                )
            if int(num_generated_tokens) >= args.max_new_tokens - 10:
                print(
                    f"  WARNING: Hit token limit ({args.max_new_tokens}). "
                    "Response may be truncated. Consider --max_new_tokens increase."
                )

        crossed_save_boundary = args.save_every <= 1 or any(
            n % args.save_every == 0 for n in range(prior_session_evaluated + 1, session_evaluated + 1)
        )
        crossed_backup_boundary = args.backup_every > 0 and any(
            n % args.backup_every == 0 for n in range(prior_session_evaluated + 1, session_evaluated + 1)
        )
        is_final_batch = batch_start + len(batch) >= len(pending_situations)
        if crossed_save_boundary or crossed_backup_boundary or is_final_batch:
            save_incremental(
                output_path,
                args,
                results,
                failed_responses,
                len(results),
                situations,
                steering_alpha=steering_alpha,
                steering_info=steering_info,
                create_backup=crossed_backup_boundary or is_final_batch,
            )

    total_elapsed = time.time() - eval_start_time
    summary_payload = summarize_result_payload(results)
    metrics = summary_payload["metrics"]
    valid = [r for r in results if r["option_type"] is not None]
    parse_failed_total = summary_payload["num_parse_failed"]
    metrics_by_subset_type = {
        **summarize_results_by_field(
            results,
            situation_manifest,
            field_name="subset_type",
            ordered_values=list(SUBSET_TYPES),
        ),
        **summarize_results_by_field(
            results,
            situation_manifest,
            field_name="probability_format",
            ordered_values=list(PROBABILITY_FORMATS),
        ),
    }
    source_stakes_metrics = summarize_results_by_field(
        results,
        situation_manifest,
        field_name="source_stakes",
        ordered_values=list(SOURCE_STAKES),
    )

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS (Permissive Parser)")
    print("=" * 50)
    print(f"Target situations: {len(situations)}")
    print(f"Completed total: {len(completed_ids)} (resumed {resumed_count}, new {session_evaluated})")
    print(f"Valid responses: {len(valid)} ({100*metrics['parse_rate']:.1f}%)")
    print(f"Failed to parse: {parse_failed_total}")
    print(f"Behaviorally classified: {summary_payload['num_behaviorally_classified']}")
    print(f"\n% choosing COOPERATE: {format_pct_metric(metrics['cooperate_rate'])}")
    print(f"% choosing REBEL:     {format_pct_metric(metrics['rebel_rate'])}")
    print(f"% choosing STEAL:     {format_pct_metric(metrics['steal_rate'])}")
    print(f"% choosing best CARA: {100*metrics['best_cara_rate']:.1f}%")
    print(f"% choosing best LIN:  {100*metrics['best_linear_rate']:.1f}%")
    avg_ev_fraction = metrics.get("avg_expected_value_fraction_of_best")
    avg_ev_relative = metrics.get("avg_expected_value_relative_to_range")
    avg_ev_regret = metrics.get("avg_expected_value_regret")
    worst_linear_rate = metrics.get("worst_linear_rate")
    if (
        avg_ev_fraction is not None
        or avg_ev_relative is not None
        or avg_ev_regret is not None
        or worst_linear_rate is not None
    ):
        if worst_linear_rate is not None:
            print(f"% choosing worst LIN: {100*worst_linear_rate:.1f}%")
        if avg_ev_fraction is not None:
            print(f"Avg EV / best EV:     {avg_ev_fraction:.3f}")
        if avg_ev_relative is not None:
            print(f"Avg EV range score:   {avg_ev_relative:.3f}")
        if avg_ev_regret is not None:
            print(f"Avg EV regret:        {avg_ev_regret:.3f}")
    if metrics_by_subset_type:
        print("\nBy subset type / probability format:")
        for group_name in list(SUBSET_TYPES) + list(PROBABILITY_FORMATS):
            subset_payload = metrics_by_subset_type.get(group_name)
            if not subset_payload:
                continue
            subset_metrics = subset_payload["metrics"]
            print(
                f"  {group_name}: valid={subset_payload['num_valid']}/{subset_payload['num_total']} | "
                f"behavioral={subset_payload['num_behaviorally_classified']} | "
                f"coop={format_pct_metric(subset_metrics['cooperate_rate'])} | "
                f"rebel={format_pct_metric(subset_metrics['rebel_rate'])} | "
                f"steal={format_pct_metric(subset_metrics['steal_rate'])} | "
                f"CARA={100*subset_metrics['best_cara_rate']:.1f}% | "
                f"LIN={100*subset_metrics['best_linear_rate']:.1f}%"
            )
    if source_stakes_metrics:
        print("\nBy source stakes:")
        for group_name in SOURCE_STAKES:
            subset_payload = source_stakes_metrics.get(group_name)
            if not subset_payload:
                continue
            subset_metrics = subset_payload["metrics"]
            line = (
                f"  {group_name}: valid={subset_payload['num_valid']}/{subset_payload['num_total']} | "
                f"bestLIN={100*subset_metrics['best_linear_rate']:.1f}% | "
                f"worstLIN={100*subset_metrics['worst_linear_rate']:.1f}%"
            )
            if subset_metrics.get("avg_expected_value_fraction_of_best") is not None:
                line += f" | EV/best={subset_metrics['avg_expected_value_fraction_of_best']:.3f}"
            if subset_metrics.get("avg_expected_value_relative_to_range") is not None:
                line += f" | EVrange={subset_metrics['avg_expected_value_relative_to_range']:.3f}"
            if subset_metrics.get("avg_expected_value_regret") is not None:
                line += f" | EVregret={subset_metrics['avg_expected_value_regret']:.3f}"
            print(line)
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    avg_per_sit = (sum(generation_times)/len(generation_times)) if generation_times else 0.0
    print(f"Avg per situation (this session): {avg_per_sit:.1f}s")
    print(
        "Avg tokens generated: "
        f"{(sum(r.get('num_tokens_generated', 0) for r in results)/len(results)) if results else 0:.0f}"
    )
    print("=" * 50)

    if failed_responses:
        print(f"\n{'='*50}")
        print(f"SAMPLE FAILED RESPONSES ({min(5, len(failed_responses))} of {len(failed_responses)})")
        print("=" * 50)
        for fr in failed_responses[:5]:
            print(f"\n--- Situation {fr['situation_id']} ({fr['num_options']} options) ---")
            print(fr["response"][:600])
            print("...")

    save_incremental(
        output_path,
        args,
        results,
        failed_responses,
        len(results),
        situations,
        steering_alpha=steering_alpha,
        steering_info=steering_info,
        create_backup=True,
    )
    print(f"\nFinal results saved to {output_path}")

    if len(completed_ids) < len(situations):
        print(
            f"Run paused with {len(situations) - len(completed_ids)} situations remaining. "
            f"Resume with: --resume --output {output_path}"
        )

    return {
        "output_path": output_path,
        "alpha": steering_alpha,
        "metrics": metrics,
        "num_valid": len(valid),
        "num_behaviorally_classified": summary_payload["num_behaviorally_classified"],
        "num_total": len(results),
        "num_parse_failed": parse_failed_total,
        "num_resumed": resumed_count,
        "num_new": session_evaluated,
    }


def make_alpha_output_path(base_output: str, alpha: float) -> str:
    """Create per-alpha output path for sweep mode."""
    p = Path(base_output)
    return str(p.with_name(f"{p.stem}_alpha_{alpha_to_suffix(alpha)}{p.suffix}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="vllm",
        help="Inference backend to use (default: vllm)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA adapter (omit to evaluate base model only)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="medium_stakes_validation",
        choices=list(DATASET_ALIASES.keys()),
        help="Built-in dataset alias (ignored if --custom_csv is provided)",
    )
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default="default",
        choices=sorted(DATASET_VARIANT_SYNONYMS.keys()),
        help=(
            "Optional built-in variant override for datasets that have separate rebels_only / steals_only / "
            "rebels_only CSV files. Prefer explicit dataset aliases like steals_test."
        ),
    )
    parser.add_argument(
        "--custom_csv",
        "--val_csv",
        dest="custom_csv",
        type=str,
        default=None,
        help="Advanced: path to custom CSV dataset (overrides --dataset).",
    )
    parser.add_argument("--list_datasets", action="store_true", help="List built-in datasets and exit")
    parser.add_argument(
        "--num_situations",
        type=int,
        default=None,
        help=(
            "Number of situations to evaluate. If omitted, Evaluate.py uses the current "
            "recommended default for the selected dataset (e.g. 200 for medium-stakes validation, "
            "1000 for the main test sets)."
        ),
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (auto-generated if omitted)")
    parser.add_argument(
        "--no_save_responses",
        action="store_true",
        help=(
            "Do NOT save full responses. Strongly discouraged for paper replication because saved responses "
            "are useful for auditing parse behavior and qualitative failures."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Max tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model ID (e.g., Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_EVAL_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_EVAL_TEMPERATURE})",
    )
    parser.add_argument(
        "--allow_nondefault_temperature",
        action="store_true",
        help=(
            "Advanced: required for any --temperature other than the canonical paper default "
            f"of {DEFAULT_EVAL_TEMPERATURE}. This prevents accidental off-default eval runs."
        ),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling cutoff (default: 0.95)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k sampling cutoff (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Sampling seed (default: 12345)",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Disable thinking mode in the chat template",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of situations to generate in parallel on one model replica (default: 4)",
    )
    parser.add_argument(
        "--max_time_per_generation",
        type=float,
        default=300,
        help="Max seconds per generation batch before timeout (default: 300)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help=(
            "Shared system prompt prepended to every situation. "
            "If omitted, evaluate.py chooses the built-in default for the selected dataset family."
        ),
    )
    parser.add_argument(
        "--force_default_system_prompt",
        action="store_true",
        help=(
            "Force the original built-in dataset system prompt into the normal system role, "
            "even for model families that otherwise default to no system prompt. "
            "Used by the steering pipeline."
        ),
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Optional extra instruction appended to each prompt before generation",
    )
    parser.add_argument(
        "--reasoning_max_tokens",
        type=int,
        default=800,
        help="Target cap for internal reasoning length, enforced via prompt instructions (default: 800)",
    )
    parser.add_argument(
        "--lin_only",
        action="store_true",
        help=(
            "Filter selected dataset slice to LIN-only situations, i.e. cases where linear-best and "
            "CARA-best labels disagree. Intended for low-stakes training/validation datasets."
        ),
    )
    parser.add_argument(
        "--start_position",
        type=int,
        default=1,
        help="1-based position in dataset order to start from (default: 1)",
    )
    parser.add_argument(
        "--end_position",
        type=int,
        default=None,
        help="1-based inclusive end position in dataset order (default: dataset end)",
    )
    parser.add_argument(
        "--stop_after",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSON if present",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=4,
        help="Write checkpoint every N newly evaluated situations (default: 4, aligned with default batch_size)",
    )
    parser.add_argument(
        "--backup_every",
        type=int,
        default=20,
        help="Write .bak backup every N newly evaluated situations (default: 20, 0 disables backups)",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM backend (default: 1)",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization target for vLLM backend (default: 0.9)",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Optional max model length override for vLLM backend",
    )
    parser.add_argument(
        "--vllm_dtype",
        type=str,
        default="auto",
        help="vLLM model dtype (default: auto)",
    )
    parser.add_argument(
        "--vllm_enable_prefix_caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable prefix caching in vLLM backend (default: on)",
    )
    parser.add_argument(
        "--vllm_max_lora_rank",
        type=int,
        default=64,
        help="Max LoRA rank for vLLM backend when --model_path is used (default: 64)",
    )

    # Steering controls (optional; defaults preserve standard evaluator behavior).
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0",
        help='Comma-separated steering strengths (e.g. "0,0.5,1.0")',
    )
    parser.add_argument(
        "--steering_direction_path",
        type=str,
        default=None,
        help="Path to a precomputed steering vector (torch tensor or dict wrapper)",
    )
    parser.add_argument(
        "--steering_apply_mode",
        choices=["last_prompt_and_current", "all_positions"],
        default="last_prompt_and_current",
        help=(
            "How to apply the steering vector on the Transformers backend. "
            "'last_prompt_and_current' steers only the last real prompt token on prefill "
            "and the current token on decode steps; 'all_positions' steers every token "
            "position in each forward pass."
        ),
    )
    parser.add_argument(
        "--eval_layer",
        type=int,
        default=None,
        help="Transformer block index (0-based) for steering injection",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("Built-in datasets (recommended current defaults):")
        for name, rel_path in CANONICAL_DATASET_ALIASES.items():
            print(f"  {name:32} -> {resolve_path(rel_path)}")
        print("\nAdditional current aliases:")
        for name, rel_path in CURRENT_EXTRA_DATASET_ALIASES.items():
            print(f"  {name:32} -> {resolve_path(rel_path)}")
        print("\nVariant overrides:")
        for dataset_name, variant_paths in DATASET_VARIANT_PATHS.items():
            variants = ", ".join(f"{variant} -> {resolve_path(path)}" for variant, path in variant_paths.items())
            print(f"  {dataset_name:32} :: {variants}")
        return

    if args.dataset in {"low_stakes_training_lin_only", "low_stakes_validation_lin_only"}:
        if not args.lin_only:
            print(f"Note: Enabling --lin_only because dataset alias {args.dataset} was selected.")
        args.lin_only = True

    if args.no_save_responses:
        print(
            "WARNING: --no_save_responses is strongly discouraged for paper replication. "
            "Keep saved responses unless you have a specific reason not to."
        )

    if args.custom_csv:
        if args.dataset != "medium_stakes_validation":
            print("Note: --custom_csv overrides --dataset; using custom dataset path.")
        if normalize_dataset_variant(args.dataset_variant) != "default":
            print("Note: --dataset_variant is ignored when --custom_csv is provided.")
        args.dataset = "custom"
        args.custom_csv = resolve_path(args.custom_csv)
        args.csv_path = args.custom_csv
        args.resolved_dataset_variant = "custom"
        args.dataset_base_alias = "custom"
    else:
        args.csv_path, args.resolved_dataset_variant, args.dataset_base_alias = resolve_builtin_dataset_path(
            args.dataset,
            args.dataset_variant,
        )

    if args.force_default_system_prompt:
        if args.system_prompt is not None:
            raise ValueError("Use either --system_prompt or --force_default_system_prompt, not both.")
        args.system_prompt = default_system_prompt_for_dataset(args.dataset_base_alias)

    args.system_prompt, args.system_prompt_source = resolve_system_prompt(
        dataset_base_alias=args.dataset_base_alias,
        base_model=args.base_model,
        model_path=args.model_path,
        explicit_system_prompt=args.system_prompt,
    )
    if args.force_default_system_prompt:
        print("Using forced original dataset system prompt in the system role.")
    elif args.system_prompt_source == DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE and args.dataset_base_alias != "custom":
        print(f"Using default system prompt for dataset family: {args.dataset_base_alias}")
    elif args.system_prompt_source == MODEL_DEFAULT_NO_SYSTEM_PROMPT_SOURCE:
        print("Using model-specific no-system-prompt default for this model family.")
    elif (
        args.system_prompt_source == CLI_SYSTEM_PROMPT_SOURCE
        and args.system_prompt.strip()
        and (model_uses_no_system_prompt(args.base_model) or model_uses_no_system_prompt(args.model_path))
    ):
        print(
            "WARNING: Gemma 3 12B runs in this repo normally use no system prompt. "
            "You overrode that with --system_prompt."
        )

    if args.lin_only and args.dataset not in {
        "custom",
        "low_stakes_training",
        "low_stakes_validation",
        "low_stakes_training_lin_only",
        "low_stakes_validation_lin_only",
    }:
        print(
            "Note: --lin_only is intended for the low-stakes training/validation datasets. "
            f"You are using it with '{args.dataset}'."
        )

    if args.dataset in {"low_stakes_validation", "low_stakes_validation_lin_only"}:
        print(
            "Note: low_stakes_validation now points to the same March 22 source CSV as low_stakes_training. "
            "Use --start_position/--end_position or --custom_csv if you want a fixed held-out validation split."
        )

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(
            f"Dataset file not found: {args.csv_path}\n"
            "Use --list_datasets to see built-in options or provide --custom_csv."
        )
    if args.start_position < 1:
        raise ValueError("--start_position must be >= 1")
    if args.end_position is not None and args.end_position < args.start_position:
        raise ValueError("--end_position must be >= --start_position")
    if args.num_situations is not None and args.num_situations < 1:
        raise ValueError("--num_situations must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")
    if args.backup_every < 0:
        raise ValueError("--backup_every must be >= 0")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("--stop_after must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if abs(args.temperature - DEFAULT_EVAL_TEMPERATURE) > 1e-12 and not args.allow_nondefault_temperature:
        raise ValueError(
            "Non-default --temperature requested "
            f"({args.temperature}). The canonical paper eval default is {DEFAULT_EVAL_TEMPERATURE}. "
            "If you really intend to run off-default, re-run with --allow_nondefault_temperature."
        )
    if not (0 < args.top_p <= 1):
        raise ValueError("--top_p must be in (0, 1]")
    if args.top_k < 0:
        raise ValueError("--top_k must be >= 0")
    if args.reasoning_max_tokens < 1:
        raise ValueError("--reasoning_max_tokens must be >= 1")
    if args.vllm_tensor_parallel_size < 1:
        raise ValueError("--vllm_tensor_parallel_size must be >= 1")
    if not (0 < args.vllm_gpu_memory_utilization <= 1):
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1]")

    alphas = parse_alpha_list(args.alphas)
    if args.backend == "vllm":
        # torch.manual_seed() seeds CUDA as well; keep vLLM parent process CPU-only until workers fork.
        torch.default_generator.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Auto-generate descriptive output filename if not provided.
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model_path:
            model_short = args.model_path.rstrip("/").split("/")[-1]
            if model_short in ("final",) or model_short.startswith("checkpoint"):
                parts = args.model_path.rstrip("/").split("/")
                model_short = parts[-2] if len(parts) >= 2 else model_short
        else:
            model_short = args.base_model.replace("/", "_") + "_base"
        args.output = f"eval_{model_short}_{args.dataset}_{args.backend}_temp{args.temperature}_{timestamp}.json"

    if args.model_path:
        print(
            f"Loading fine-tuned model (backend: {args.backend}, "
            f"base: {args.base_model}, adapter: {args.model_path})..."
        )
    else:
        print(f"Loading base model only (backend: {args.backend}): {args.base_model}")

    if args.backend == "vllm":
        # Steering needs eager mode (so the worker hook fires) and prefix caching OFF
        # (the cache keys on token ids and would reuse unsteered KV across alphas).
        # Set both on args before the engine build so logs + recorded config are truthful.
        args._vllm_enforce_eager = bool(args.steering_direction_path)
        if args._vllm_enforce_eager:
            args.vllm_enable_prefix_caching = False
        model, lora_request = load_vllm_engine(args)
        tokenizer = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if args.model_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, args.model_path)
            model = model.merge_and_unload()
        else:
            model = base_model

        model.eval()
        lora_request = None

    print("Loading validation data...")
    df = pd.read_csv(args.csv_path)
    df = ensure_option_level_dataframe(df)
    validate_dataset_columns(df, args.csv_path)
    print(f"Dataset alias: {args.dataset}")
    print(f"Dataset base alias: {args.dataset_base_alias}")
    print(f"Dataset variant: {args.resolved_dataset_variant}")
    print(f"Dataset path:  {args.csv_path}")
    if args.num_situations is None:
        default_num_situations = resolve_default_num_situations(args)
        if default_num_situations is None:
            default_num_situations = int(df["situation_id"].nunique())
            print(
                f"Num situations not specified; defaulting to full selected dataset "
                f"({default_num_situations} situations)."
            )
        else:
            print(
                f"Num situations not specified; defaulting to the recommended current setting "
                f"for {args.dataset}: {default_num_situations}."
            )
        args.num_situations = default_num_situations
    all_situations = build_situations(df, args.num_situations)

    end_position = args.end_position if args.end_position is not None else len(all_situations)
    if args.start_position > len(all_situations):
        raise ValueError(
            f"--start_position ({args.start_position}) is beyond available situations ({len(all_situations)})."
        )
    situations = all_situations[args.start_position - 1 : end_position]
    args.end_position = end_position
    if args.lin_only:
        before_lin_filter = len(situations)
        situations = filter_lin_only_situations(situations)
        print(f"LIN-only filter active: kept {len(situations)}/{before_lin_filter} situations in selected slice.")
    if not situations:
        raise ValueError("No situations selected after applying --start_position/--end_position.")
    print(
        f"Selected situation positions: {args.start_position}.."
        f"{args.start_position + len(situations) - 1} (count={len(situations)})"
    )

    steering_direction = None
    steering_block = None
    steering_info = None
    layers = None
    n_layers = None

    if args.backend == "vllm":
        # vLLM steering (all-positions / CAA) is applied via an in-process forward
        # hook (see steering/vllm_steering.py). Build the direction first with
        # steering/build_steering_direction.py and pass it via
        # --steering_direction_path.
        if args.steering_direction_path or any(abs(alpha) > 0 for alpha in alphas):
            from vllm_steering import get_vllm_n_layers

            n_layers = get_vllm_n_layers(model)
    else:
        layers = get_decoder_layers(model)
        n_layers = len(layers)

    nonzero_alphas = [a for a in alphas if abs(a) > 0]
    if nonzero_alphas and args.steering_direction_path is None:
        raise ValueError(
            "Non-zero --alphas requires --steering_direction_path."
        )

    if args.steering_direction_path:
        steering_direction, steering_artifact_info = load_steering_artifact(args.steering_direction_path)
        saved_layer = steering_artifact_info.get("eval_layer")
        if saved_layer is None:
            saved_layer = steering_artifact_info.get("extraction_layer")
        if saved_layer is None:
            saved_layer = steering_artifact_info.get("layer")
        eval_layer = args.eval_layer if args.eval_layer is not None else saved_layer
        if eval_layer is None:
            eval_layer = n_layers // 2
        eval_layer = int(eval_layer)
        if not (0 <= eval_layer < n_layers):
            raise ValueError(f"--eval_layer out of range: {eval_layer}, model has {n_layers} layers")
        if args.backend == "vllm":
            # All-positions CAA hook on the live vLLM engine; alpha is set per run
            # inside run_single_alpha_eval via model._steering_controller.
            from vllm_steering import attach_vllm_steering

            apply_mode = "all_positions"
            model._steering_controller = attach_vllm_steering(
                model, steering_direction, eval_layer, alpha=0.0, apply_mode=apply_mode,
            )
        else:
            apply_mode = args.steering_apply_mode
            steering_block = layers[eval_layer]
        steering_info = {
            "mode": "precomputed_vector",
            "vector_path": args.steering_direction_path,
            "eval_layer": eval_layer,
            "apply_mode": apply_mode,
            "direction_norm": float(steering_direction.norm(p=2).item()),
            "artifact_metadata": convert_numpy(steering_artifact_info),
        }

    per_alpha_summaries = []
    multi_alpha = len(alphas) > 1

    for alpha in alphas:
        print("\n" + "=" * 72)
        print(f"Running evaluation for alpha={alpha:+.4f}")
        print("=" * 72)

        alpha_output = make_alpha_output_path(args.output, alpha) if multi_alpha else args.output

        summary = run_single_alpha_eval(
            backend=args.backend,
            model=model,
            tokenizer=tokenizer,
            situations=situations,
            args=args,
            output_path=alpha_output,
            steering_alpha=alpha,
            steering_info=steering_info,
            steering_block=steering_block,
            steering_direction=steering_direction,
            lora_request=lora_request,
        )
        per_alpha_summaries.append(summary)

    if multi_alpha:
        selected_situations = build_situation_manifest(situations)
        selected_subset_type_counts = {
            subset_type: sum(1 for entry in selected_situations if entry.get("subset_type") == subset_type)
            for subset_type in SUBSET_TYPES
            if any(entry.get("subset_type") == subset_type for entry in selected_situations)
        }
        sweep_payload = convert_numpy(
            {
                "evaluation_config": {
                    "backend": args.backend,
                    "base_model": args.base_model,
                    "model_path": args.model_path,
                    "dataset": args.dataset,
                    "dataset_base_alias": args.dataset_base_alias,
                    "dataset_variant": args.resolved_dataset_variant,
                    "custom_csv": args.custom_csv,
                    "csv_path": args.csv_path,
                    "num_situations": len(situations),
                    "start_position": args.start_position,
                    "end_position": end_position,
                    "lin_only": args.lin_only,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "seed": args.seed,
                    "max_new_tokens": args.max_new_tokens,
                    "reasoning": {"max_tokens": args.reasoning_max_tokens},
                    "batch_size": args.batch_size,
                    "max_time_per_generation": args.max_time_per_generation,
                    "system_prompt": args.system_prompt,
                    "prompt_suffix": args.prompt_suffix,
                    "enable_thinking": not args.disable_thinking,
                    "steering_apply_mode": args.steering_apply_mode,
                    "alphas": alphas,
                    "resume": args.resume,
                    "save_every": args.save_every,
                    "backup_every": args.backup_every,
                    "stop_after": args.stop_after,
                    "selected_situation_ids": [sit["situation_id"] for sit in situations],
                    "selected_subset_type_counts": selected_subset_type_counts,
                    "selected_situations": selected_situations,
                    "steering": steering_info,
                },
                "runs": per_alpha_summaries,
            }
        )
        if args.backend == "vllm":
            sweep_payload["evaluation_config"]["vllm"] = vllm_settings_from_args(args)
        with open(args.output, "w") as f:
            json.dump(sweep_payload, f, indent=2)
        print(f"\nSweep summary saved to {args.output}")
        print("Per-alpha outputs:")
        for run in per_alpha_summaries:
            print(f"  alpha={run['alpha']:+.4f} -> {run['output_path']}")

    del model
    gc.collect()
    if args.backend != "vllm" and torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
