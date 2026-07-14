#!/usr/bin/env python3
"""Evaluate local Hugging Face reward models on pairwise preference data."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from risk_averse_prompts import (
    CLI_SYSTEM_PROMPT_SOURCE,
    DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE,
    MODEL_DEFAULT_NO_SYSTEM_PROMPT_SOURCE,
    model_uses_no_system_prompt,
    resolve_system_prompt,
)
from cot_csv_utils import format_summary, summarize_cot_dataframe, validate_no_literal_backslash_newlines

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in non-ML envs
    torch = None

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - exercised only in non-ML envs
    PeftModel = None

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - exercised only in non-ML envs
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


# Flush output immediately so logs are visible in real time.
sys.stdout.reconfigure(line_buffering=True)
if torch is not None and torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANONICAL_DATASET_ALIASES = {
    "reward_model_validation": "data/2026_03_22_reward_model_val_set_400_Rebels_clean.csv",
}
CURRENT_EXTRA_DATASET_ALIASES = {
    "reward_model_validation_rebels_only": "data/2026_03_22_reward_model_val_set_400_Rebels_clean.csv",
    "reward_model_high_stakes_test": "data/2026_03_22_high_stakes_test_set_746_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
    "reward_model_high_stakes_test_rebels_only": "data/2026_03_22_high_stakes_test_set_746_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
    "reward_model_astronomical_stakes_deployment": "data/2026_03_22_astronomical_stakes_deployment_set_707_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
    "reward_model_astronomical_stakes_deployment_rebels_only": "data/2026_03_22_astronomical_stakes_deployment_set_707_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
    "reward_model_steals_test": "data/2026_03_22_test_set_928_Steals_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
    "reward_model_steals_test_steals_only": "data/2026_03_22_test_set_928_Steals_CoTs_for_evaluating_reward_model_from_Sonnet.csv",
}
DATASET_ALIASES = {
    **CANONICAL_DATASET_ALIASES,
    **CURRENT_EXTRA_DATASET_ALIASES,
}
REQUIRED_COLUMNS = {"prompt_text", "chosen_full", "rejected_full"}
SUBSET_TYPES = ("rebels_only", "steals_only")
PROBABILITY_FORMATS = ("numerical", "verbal")


def resolve_path(path: str) -> str:
    """Resolve a path relative to either this script or the current working directory."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    script_relative = os.path.abspath(os.path.join(SCRIPT_DIR, expanded))
    if os.path.exists(script_relative):
        return script_relative
    return os.path.abspath(expanded)


def count_dataset_rows(csv_path: str) -> Optional[int]:
    """Best-effort row count for list_datasets output."""
    try:
        return len(pd.read_csv(csv_path))
    except Exception:
        return None


def remove_instruction_suffix(prompt: str) -> str:
    """Remove old evaluation instructions that were embedded in prompt text."""
    patterns = [
        r"\s*You can think before answering,.*?would select\.",
        r"\s*You can think.*?must finish with.*?\.",
    ]
    out = prompt
    for pattern in patterns:
        out = re.sub(pattern, "", out, flags=re.IGNORECASE | re.DOTALL)
    return out.strip()


def build_eval_prompt(prompt_raw: str, prompt_suffix: str) -> str:
    """Normalize the dataset prompt and append an optional suffix."""
    prompt = remove_instruction_suffix(prompt_raw)
    return f"{prompt}\n\n{prompt_suffix}".strip() if prompt_suffix else prompt


def infer_probability_format(prompt_text: Optional[str]) -> Optional[str]:
    """Infer whether a prompt uses verbal or numerical probabilities."""
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return None
    verbal_markers = [
        "half-chance",
        "unlikely",
        "likely",
        "probable",
        "improbable",
        "highly likely",
        "highly unlikely",
        "almost certainly",
        "very likely",
        "very unlikely",
        "somewhat likely",
        "somewhat unlikely",
        "chance of",
    ]
    prompt_lower = prompt_text.lower()
    if any(marker in prompt_lower for marker in verbal_markers):
        return "verbal"
    return "numerical"


def clean_subset_type(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    subset_type = str(value).strip().lower().replace("-", "_")
    if subset_type in {"lin", "rebel_cooperate", "rebels_only"}:
        return "rebels_only"
    if subset_type in {"too_risk", "steal_mixed", "with_steals", "steals_only"}:
        return "steals_only"
    return subset_type


def clean_optional_text(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text if text else None


def clean_optional_int(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except Exception:
        return None


def validate_dataset_columns(df: pd.DataFrame, dataset_path: str):
    """Validate that the dataset has the minimum schema needed for RM evaluation."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Dataset path: {dataset_path}"
        )


def convert_numpy(obj: Any):
    """Convert numpy/pytorch scalar-like types for JSON serialization."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


def atomic_write_json(path: str, payload: Dict[str, Any]):
    """Write JSON atomically to reduce corruption risk on interruption."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, output_path)


def get_input_device(model) -> torch.device:
    """Best-effort device for tokenizer tensors when model may be sharded."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def parse_torch_dtype(dtype_name: str):
    """Map CLI dtype string to torch dtype understood by from_pretrained."""
    name = str(dtype_name).strip().lower()
    if name == "auto":
        return "auto"
    if torch is None:
        raise ImportError("torch is required to load reward models.")
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[name]


def is_batch_size_related_runtime_error(exc: BaseException) -> bool:
    """Best-effort check for runtime failures where lowering batch size often helps."""
    message = str(exc).lower()
    markers = [
        "out of memory",
        "cuda out of memory",
        "resource exhausted",
        "resource_exhausted",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "mps backend out of memory",
        "hip out of memory",
        "alloc failed",
        "xla runtime error",
    ]
    return any(marker in message for marker in markers)


def print_batch_size_troubleshooting_hint(batch_size: int):
    """Print a short actionable note when scoring likely failed from memory pressure."""
    lower_batch_size = max(batch_size // 2, 1)
    print(
        "\nBatch-size troubleshooting hint:\n"
        "  This failure looks like memory or resource exhaustion during reward-model scoring.\n"
        f"  Try rerunning with a smaller --batch_size, for example --batch_size {lower_batch_size}.\n"
        "  If the problem persists, keep the same output path and add --resume."
    )


def build_messages(prompt: str, response: str, system_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return messages


def fallback_plain_chat(messages: List[Dict[str, str]]) -> str:
    """Plain-text fallback when tokenizer chat templates are unavailable."""
    blocks = []
    for message in messages:
        role = str(message["role"]).strip().capitalize()
        blocks.append(f"{role}:\n{message['content']}".strip())
    return "\n\n".join(blocks).strip()


def format_pair_text(
    tokenizer,
    *,
    prompt: str,
    response: str,
    system_prompt: str,
    format_mode: str,
) -> str:
    """Format prompt/response for the reward model."""
    messages = build_messages(prompt, response, system_prompt)
    if format_mode == "plain_text":
        return fallback_plain_chat(messages)

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            if format_mode == "chat_template":
                raise
            return fallback_plain_chat(messages)
    except Exception:
        if format_mode == "chat_template":
            raise
        return fallback_plain_chat(messages)


def extract_reward_scores(outputs: Any, reward_output_index: Optional[int]) -> torch.Tensor:
    """Extract a single scalar reward from model outputs."""
    tensor = None
    for key in ("logits", "scores", "end_scores", "reward", "rewards"):
        if isinstance(outputs, dict) and key in outputs:
            tensor = outputs[key]
            break
        if hasattr(outputs, key):
            tensor = getattr(outputs, key)
            break

    if tensor is None:
        raise ValueError(
            "Could not find reward scores in model outputs. Expected one of logits/scores/end_scores/reward(s)."
        )

    if not torch.is_tensor(tensor):
        tensor = torch.tensor(tensor)

    if tensor.ndim == 0:
        return tensor.reshape(1).to(torch.float32)
    if tensor.ndim == 1:
        return tensor.to(torch.float32)
    if tensor.ndim == 2 and tensor.shape[-1] == 1:
        return tensor[:, 0].to(torch.float32)
    if tensor.ndim == 2 and reward_output_index is not None:
        if not (0 <= reward_output_index < tensor.shape[-1]):
            raise ValueError(
                f"--reward_output_index {reward_output_index} is out of range for reward head width {tensor.shape[-1]}"
            )
        return tensor[:, reward_output_index].to(torch.float32)

    raise ValueError(
        f"Reward model returned tensor with shape {tuple(tensor.shape)}. "
        "Provide --reward_output_index if the model has multiple output heads."
    )


class RftRewardModelWrapper(torch.nn.Module if torch is not None else object):
    """AutoModel + LoRA + separate fp32 reward head, exposed to the eval loop.

    Mirrors the forward path in rft_pipeline.py: right-padded inputs, last
    non-pad hidden state, cast to the head's dtype (fp32), then a single
    nn.Linear(hidden_size, 1, bias=True). Returns a dict with "logits" so
    extract_reward_scores finds the scalar reward unchanged.
    """

    def __init__(self, backbone, reward_head, config):
        super().__init__()
        self.backbone = backbone
        self.reward_head = reward_head
        self.config = config

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        batch_size = hidden.shape[0]
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=-1) - 1
        else:
            seq_lens = torch.full(
                (batch_size,), hidden.shape[1] - 1, dtype=torch.long, device=hidden.device
            )
        last = hidden[torch.arange(batch_size, device=hidden.device), seq_lens]
        last = last.to(self.reward_head.weight.dtype)
        rewards = self.reward_head(last)
        return {"logits": rewards}


def build_reward_head_from_checkpoint(reward_head_path: str):
    """Return an nn.Linear(hidden_size, 1) matching an rft_pipeline.py checkpoint.

    The checkpoint format is `{"reward_head_state_dict": <state>, "hidden_size": int}`
    (see rft_pipeline.py:470-473). Weights are kept in fp32 to match training.
    """
    if torch is None:
        raise ImportError("torch is required to load a reward head checkpoint.")
    if not os.path.exists(reward_head_path):
        raise FileNotFoundError(f"reward_head checkpoint not found: {reward_head_path}")
    ckpt = torch.load(reward_head_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "reward_head_state_dict" not in ckpt:
        raise ValueError(
            f"{reward_head_path} is not an rft_pipeline reward_head.pt "
            "(expected key 'reward_head_state_dict')."
        )
    state = ckpt["reward_head_state_dict"]
    if "weight" not in state:
        raise ValueError(f"{reward_head_path} state dict missing 'weight' tensor.")
    out_features, in_features = state["weight"].shape
    has_bias = "bias" in state
    head = torch.nn.Linear(in_features, out_features, bias=has_bias).to(torch.float32)
    head.load_state_dict(state)
    return head, int(ckpt.get("hidden_size", in_features))


def resolve_reward_head_path(args) -> Optional[str]:
    """Return the reward_head.pt path to load, or None if we should skip it.

    Precedence: explicit --reward_head_path wins (empty string disables),
    then auto-detect reward_head.pt inside --model_path.
    """
    explicit = getattr(args, "reward_head_path", None)
    if explicit is not None:
        if explicit == "":
            return None
        return resolve_path(explicit)
    if args.model_path:
        candidate = os.path.join(args.model_path, "reward_head.pt")
        if os.path.exists(candidate):
            return candidate
    return None


def load_reward_model(args):
    """Load tokenizer + reward model, optionally with a PEFT adapter.

    Two loader paths:
      1. rft_pipeline.py format (default when reward_head.pt is present next
         to --model_path, or --reward_head_path is provided): AutoModel +
         LoRA adapter + separate fp32 reward head. See RftRewardModelWrapper.
      2. Standard HF reward model: AutoModelForSequenceClassification, with
         the score head from the checkpoint. Optional PEFT adapter via
         --model_path (the adapter target paths must match this wrapper).
    """
    if torch is None or AutoModelForSequenceClassification is None or AutoTokenizer is None:
        raise ImportError(
            "Reward model evaluation requires torch, transformers, and peft to be installed "
            "in the current environment."
        )
    load_source = args.base_model or args.model_path
    if not load_source:
        raise ValueError("Provide either --base_model, --model_path, or both.")

    tokenizer_source = args.tokenizer_name or load_source
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        use_fast=not args.no_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    reward_head_path = resolve_reward_head_path(args)
    if reward_head_path is not None:
        if AutoModel is None:
            raise ImportError("transformers AutoModel is required to load rft_pipeline checkpoints.")
        print(f"[load_reward_model] loading rft_pipeline checkpoint with reward_head={reward_head_path}")
        backbone = AutoModel.from_pretrained(
            load_source,
            torch_dtype=parse_torch_dtype(args.torch_dtype),
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
        if args.base_model and args.model_path:
            if PeftModel is None:
                raise ImportError("peft is required when using --model_path with a LoRA adapter.")
            backbone = PeftModel.from_pretrained(backbone, args.model_path)
        if getattr(backbone.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            backbone.config.pad_token_id = tokenizer.pad_token_id
        reward_head, ckpt_hidden = build_reward_head_from_checkpoint(reward_head_path)
        model_hidden = getattr(backbone.config, "hidden_size", None)
        if model_hidden is not None and ckpt_hidden != model_hidden:
            raise ValueError(
                f"reward_head.pt hidden_size={ckpt_hidden} does not match "
                f"backbone hidden_size={model_hidden}"
            )
        reward_head = reward_head.to(next(backbone.parameters()).device)
        model = RftRewardModelWrapper(backbone, reward_head, backbone.config)
        model.eval()
        return model, tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        load_source,
        # A reward model has a single scalar score head (num_labels=1). Without this, HF defaults
        # to num_labels=2 -> score[2, hidden], which mismatches a num_labels=1 reward adapter on
        # load (RuntimeError: size mismatch for ...score...weight [1, h] vs [2, h]).
        num_labels=1,
        torch_dtype=parse_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.base_model and args.model_path:
        if PeftModel is None:
            raise ImportError("peft is required when using --model_path with a reward model adapter.")
        model = PeftModel.from_pretrained(model, args.model_path)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def build_pair_manifest_entry(pair: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "dataset_position": pair["dataset_position"],
        "situation_id": pair.get("situation_id"),
        "subset_type": pair.get("subset_type"),
        "probability_format": pair.get("probability_format"),
        "accepted_output_tokens": pair.get("accepted_output_tokens"),
        "rejected_output_tokens": pair.get("rejected_output_tokens"),
    }


def build_pair_manifest(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [build_pair_manifest_entry(pair) for pair in pairs]


def build_pair_manifest_index(pairs: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {entry["pair_id"]: entry for entry in build_pair_manifest(pairs)}


def annotate_rows_with_pair_metadata(rows: List[Dict[str, Any]], pair_index: Dict[int, Dict[str, Any]]):
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id is None:
            continue
        meta = pair_index.get(pair_id, {})
        for key in ("dataset_position", "situation_id", "subset_type", "probability_format"):
            if key == "subset_type":
                row[key] = meta.get(key)
                continue
            if row.get(key) is None:
                row[key] = meta.get(key)


def compute_length_relation(accepted_tokens: Optional[int], rejected_tokens: Optional[int]) -> Optional[str]:
    if accepted_tokens is None or rejected_tokens is None:
        return None
    if accepted_tokens > rejected_tokens:
        return "accepted_longer"
    if rejected_tokens > accepted_tokens:
        return "rejected_longer"
    return "same_length"


def approximate_pair_length(pair: Dict[str, Any]) -> int:
    """Cheap proxy for batching similar-length prompt/response pairs together."""
    prompt_len = len(pair.get("prompt", "") or pair.get("prompt_raw", "") or "")
    accepted_len = pair.get("accepted_output_tokens")
    rejected_len = pair.get("rejected_output_tokens")
    if accepted_len is not None or rejected_len is not None:
        return prompt_len + max(accepted_len or 0, rejected_len or 0)
    return prompt_len + max(len(pair.get("accepted_response", "")), len(pair.get("rejected_response", "")))


def predict_preference(accepted_score: float, rejected_score: float, tie_epsilon: float) -> str:
    margin = accepted_score - rejected_score
    if margin > tie_epsilon:
        return "accepted"
    if margin < -tie_epsilon:
        return "rejected"
    return "tie"


def logistic_loss_from_margin(margin: float) -> float:
    """Pairwise Bradley-Terry style loss for a chosen-vs-rejected margin."""
    if margin >= 0:
        return math.log1p(math.exp(-margin))
    return -margin + math.log1p(math.exp(margin))


def safe_rate(rows: List[Dict[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if predicate(row)) / len(rows)


def summarize_pairwise_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    margins = [float(row["score_margin"]) for row in results]
    accepted_scores = [float(row["accepted_score"]) for row in results]
    rejected_scores = [float(row["rejected_score"]) for row in results]
    ties = [row for row in results if row.get("predicted_preference") == "tie"]
    correct = [row for row in results if row.get("is_correct") is True]
    ties_half_credit = [
        1.0 if row.get("predicted_preference") == "accepted" else 0.5 if row.get("predicted_preference") == "tie" else 0.0
        for row in results
    ]
    truncated_pairs = [
        row for row in results if bool(row.get("accepted_truncated")) or bool(row.get("rejected_truncated"))
    ]
    with_length_data = [row for row in results if row.get("length_relation") is not None]
    accepted_longer = [row for row in with_length_data if row.get("length_relation") == "accepted_longer"]
    rejected_longer = [row for row in with_length_data if row.get("length_relation") == "rejected_longer"]
    same_length = [row for row in with_length_data if row.get("length_relation") == "same_length"]

    metrics = {
        "pairwise_accuracy": safe_rate(results, lambda row: row.get("is_correct") is True),
        "pairwise_accuracy_ties_half_credit": (sum(ties_half_credit) / len(ties_half_credit)) if ties_half_credit else 0.0,
        "tie_rate": safe_rate(results, lambda row: row.get("predicted_preference") == "tie"),
        "preference_log_loss": (sum(logistic_loss_from_margin(m) for m in margins) / len(margins)) if margins else 0.0,
        "mean_score_margin": (sum(margins) / len(margins)) if margins else 0.0,
        "median_score_margin": median(margins) if margins else 0.0,
        "mean_abs_score_margin": (sum(abs(m) for m in margins) / len(margins)) if margins else 0.0,
        "mean_accepted_score": (sum(accepted_scores) / len(accepted_scores)) if accepted_scores else 0.0,
        "mean_rejected_score": (sum(rejected_scores) / len(rejected_scores)) if rejected_scores else 0.0,
        "accepted_truncated_rate": safe_rate(results, lambda row: bool(row.get("accepted_truncated"))),
        "rejected_truncated_rate": safe_rate(results, lambda row: bool(row.get("rejected_truncated"))),
        "truncated_pair_rate": safe_rate(results, lambda row: bool(row.get("accepted_truncated")) or bool(row.get("rejected_truncated"))),
        "pairwise_accuracy_when_accepted_longer": safe_rate(accepted_longer, lambda row: row.get("is_correct") is True),
        "pairwise_accuracy_when_rejected_longer": safe_rate(rejected_longer, lambda row: row.get("is_correct") is True),
        "pairwise_accuracy_when_same_length": safe_rate(same_length, lambda row: row.get("is_correct") is True),
    }

    return {
        "metrics": metrics,
        "num_total": len(results),
        "num_correct": len(correct),
        "num_incorrect": len(results) - len(correct) - len(ties),
        "num_ties": len(ties),
        "num_truncated_pairs": len(truncated_pairs),
        "num_pairs_with_length_data": len(with_length_data),
    }


def summarize_results_by_field(
    results: List[Dict[str, Any]],
    pair_manifest: List[Dict[str, Any]],
    field_name: str,
    ordered_values: Optional[List[str]] = None,
) -> Dict[str, Any]:
    values_in_target = []
    for entry in pair_manifest:
        value = entry.get(field_name)
        if value is None or value in values_in_target:
            continue
        values_in_target.append(value)
    if ordered_values is not None:
        ordered = [value for value in ordered_values if value in values_in_target]
        ordered.extend(value for value in values_in_target if value not in ordered)
        values_in_target = ordered

    summarized = {}
    for value in values_in_target:
        subset_results = [row for row in results if row.get(field_name) == value]
        summarized[value] = summarize_pairwise_results(subset_results)
    return summarized


def summarize_progress_by_subset_type(results: List[Dict[str, Any]], pair_manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed_ids = {row.get("pair_id") for row in results if row.get("pair_id") is not None}
    progress = {}
    for subset_type in SUBSET_TYPES:
        target_ids = [entry["pair_id"] for entry in pair_manifest if entry.get("subset_type") == subset_type]
        if not target_ids:
            continue
        completed = sum(1 for pid in target_ids if pid in completed_ids)
        next_pair_id = next((pid for pid in target_ids if pid not in completed_ids), None)
        progress[subset_type] = {
            "target_total": len(target_ids),
            "completed": completed,
            "remaining": max(len(target_ids) - completed, 0),
            "next_pair_id": next_pair_id,
        }
    return progress


def project_result_row_for_output(row: Dict[str, Any], *, include_responses: bool) -> Dict[str, Any]:
    keys = [
        "pair_id",
        "dataset_position",
        "situation_id",
        "subset_type",
        "probability_format",
        "prompt",
        "accepted_expected",
        "rejected_expected",
        "accepted_output_tokens",
        "rejected_output_tokens",
        "accepted_stop_reason",
        "rejected_stop_reason",
        "accepted_score",
        "rejected_score",
        "score_margin",
        "predicted_preference",
        "is_correct",
        "length_relation",
        "accepted_input_length",
        "rejected_input_length",
        "accepted_truncated",
        "rejected_truncated",
        "scoring_batch_time_seconds",
        "scoring_batch_size",
    ]
    projected = {key: row.get(key) for key in keys}
    if include_responses:
        projected["accepted_response"] = row.get("accepted_response")
        projected["rejected_response"] = row.get("rejected_response")
    return projected


def compact_results_for_resume(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [project_result_row_for_output(row, include_responses=False) for row in results]


def dedupe_results_by_pair_id(results: List[Dict[str, Any]], ordered_pair_ids: List[int]) -> List[Dict[str, Any]]:
    latest_by_id = {}
    for row in results:
        pair_id = row.get("pair_id")
        if pair_id is None:
            continue
        latest_by_id[pair_id] = row
    return [latest_by_id[pair_id] for pair_id in ordered_pair_ids if pair_id in latest_by_id]


def load_existing_run_state(output_path: str, ordered_pair_ids: List[int], *, allow_backup_fallback: bool = True):
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
            raise RuntimeError(f"Found prior output but failed to parse JSON: {output_path} ({last_error})") from last_error
        return None

    results = loaded.get("results")
    if not isinstance(results, list):
        results = loaded.get("resume_records")
    if not isinstance(results, list):
        raise ValueError(
            "Cannot resume: output JSON does not contain resumable records. "
            "Expected `results` or `resume_records` as a list."
        )

    deduped_results = dedupe_results_by_pair_id(results, ordered_pair_ids)
    rows_in_target = [row for row in results if row.get("pair_id") in set(ordered_pair_ids)]
    dropped_duplicates = max(len(rows_in_target) - len(deduped_results), 0)

    return {
        "loaded_from": loaded_from,
        "payload": loaded,
        "results": deduped_results,
        "dropped_duplicates": dropped_duplicates,
    }


def build_preference_pairs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for idx, row in enumerate(df.to_dict("records"), start=1):
        prompt_raw = str(row["prompt_text"])
        accepted_tokens = clean_optional_int(row.get("chosen_output_tokens"))
        rejected_tokens = clean_optional_int(row.get("rejected_output_tokens"))
        pair = {
            "pair_id": idx,
            "dataset_position": idx,
            "situation_id": clean_optional_int(row.get("situation_id")),
            "subset_type": clean_subset_type(row.get("subset_type") if "subset_type" in row else row.get("rejected_type")),
            "prompt_raw": prompt_raw,
            "probability_format": infer_probability_format(prompt_raw),
            "accepted_expected": clean_optional_text(row.get("chosen_expected")),
            "rejected_expected": clean_optional_text(row.get("rejected_expected")),
            "accepted_response": str(row["chosen_full"]),
            "rejected_response": str(row["rejected_full"]),
            "accepted_output_tokens": accepted_tokens,
            "rejected_output_tokens": rejected_tokens,
            "accepted_stop_reason": clean_optional_text(row.get("chosen_stop_reason")),
            "rejected_stop_reason": clean_optional_text(row.get("rejected_stop_reason")),
            "length_relation": compute_length_relation(accepted_tokens, rejected_tokens),
        }
        pairs.append(pair)
    return pairs


def select_pairs(
    pairs: List[Dict[str, Any]],
    *,
    start_position: int,
    end_position: Optional[int],
    num_pairs: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    sliced = [pair for pair in pairs if pair["dataset_position"] >= start_position]
    if end_position is not None:
        sliced = [pair for pair in sliced if pair["dataset_position"] <= end_position]

    selected: List[Dict[str, Any]] = []
    seen_exact_rows: Set[Tuple[str, str, str]] = set()
    exact_duplicate_rows_skipped = 0

    for pair in sliced:
        dedupe_key = (
            str(pair.get("prompt_raw", "")),
            str(pair.get("accepted_response", "")),
            str(pair.get("rejected_response", "")),
        )
        if dedupe_key in seen_exact_rows:
            exact_duplicate_rows_skipped += 1
            continue
        seen_exact_rows.add(dedupe_key)
        selected.append(pair)

        if num_pairs is not None and len(selected) >= num_pairs:
            break

    selection_stats = {
        "raw_pair_rows_in_slice": len(sliced),
        "selected_unique_rows": len(selected),
        "exact_duplicate_rows_skipped": exact_duplicate_rows_skipped,
    }
    return selected, selection_stats


def ensure_output_path_is_safe(output_path: str, *, resume: bool):
    if resume:
        return
    if Path(output_path).exists():
        raise FileExistsError(
            "Output file already exists. To continue the interrupted run, re-run with "
            f"--resume --output {output_path}. To start fresh, choose a new --output path "
            "or delete the old output file first."
        )


def score_batch(
    *,
    model,
    tokenizer,
    batch_pairs: List[Dict[str, Any]],
    system_prompt: str,
    format_mode: str,
    max_length: int,
    reward_output_index: Optional[int],
) -> List[Dict[str, Any]]:
    accepted_texts = [
        format_pair_text(
            tokenizer,
            prompt=pair["prompt"],
            response=pair["accepted_response"],
            system_prompt=system_prompt,
            format_mode=format_mode,
        )
        for pair in batch_pairs
    ]
    rejected_texts = [
        format_pair_text(
            tokenizer,
            prompt=pair["prompt"],
            response=pair["rejected_response"],
            system_prompt=system_prompt,
            format_mode=format_mode,
        )
        for pair in batch_pairs
    ]
    all_texts = accepted_texts + rejected_texts

    raw_tokenized = tokenizer(all_texts, padding=False, truncation=False, add_special_tokens=True)
    raw_lengths = [len(ids) for ids in raw_tokenized["input_ids"]]

    encoded = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = get_input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    batch_start = time.time()
    with torch.inference_mode():
        outputs = model(**encoded)
        scores = extract_reward_scores(outputs, reward_output_index).detach().cpu().tolist()
    batch_elapsed = round(time.time() - batch_start, 2)

    midpoint = len(batch_pairs)
    results = []
    for idx, pair in enumerate(batch_pairs):
        accepted_score = float(scores[idx])
        rejected_score = float(scores[midpoint + idx])
        accepted_length = int(raw_lengths[idx])
        rejected_length = int(raw_lengths[midpoint + idx])
        results.append(
            {
                "pair_id": pair["pair_id"],
                "dataset_position": pair["dataset_position"],
                "situation_id": pair.get("situation_id"),
                "subset_type": pair.get("subset_type"),
                "probability_format": pair.get("probability_format"),
                "prompt": pair["prompt"],
                "accepted_expected": pair.get("accepted_expected"),
                "rejected_expected": pair.get("rejected_expected"),
                "accepted_output_tokens": pair.get("accepted_output_tokens"),
                "rejected_output_tokens": pair.get("rejected_output_tokens"),
                "accepted_stop_reason": pair.get("accepted_stop_reason"),
                "rejected_stop_reason": pair.get("rejected_stop_reason"),
                "accepted_response": pair["accepted_response"],
                "rejected_response": pair["rejected_response"],
                "accepted_score": accepted_score,
                "rejected_score": rejected_score,
                "score_margin": accepted_score - rejected_score,
                "predicted_preference": None,  # filled later
                "is_correct": None,  # filled later
                "length_relation": pair.get("length_relation"),
                "accepted_input_length": accepted_length,
                "rejected_input_length": rejected_length,
                "accepted_truncated": accepted_length > max_length,
                "rejected_truncated": rejected_length > max_length,
                "scoring_batch_time_seconds": batch_elapsed,
                "scoring_batch_size": len(batch_pairs),
            }
        )
    return results


def save_incremental(
    output_path: str,
    args,
    results: List[Dict[str, Any]],
    target_pairs: List[Dict[str, Any]],
    *,
    selection_stats: Optional[Dict[str, int]] = None,
    create_backup: bool = False,
):
    pair_manifest = build_pair_manifest(target_pairs)
    pair_index = build_pair_manifest_index(target_pairs)
    annotate_rows_with_pair_metadata(results, pair_index)
    target_pair_ids = [entry["pair_id"] for entry in pair_manifest]
    ordered_results = dedupe_results_by_pair_id(results, target_pair_ids)

    summary_payload = summarize_pairwise_results(ordered_results)
    done_ids = {row.get("pair_id") for row in ordered_results if row.get("pair_id") is not None}
    target_total = len(target_pair_ids)
    target_completed = sum(1 for pair_id in target_pair_ids if pair_id in done_ids)
    next_pair_id = next((pair_id for pair_id in target_pair_ids if pair_id not in done_ids), None)
    selected_subset_type_counts = {
        subset_type: sum(1 for entry in pair_manifest if entry.get("subset_type") == subset_type)
        for subset_type in SUBSET_TYPES
        if any(entry.get("subset_type") == subset_type for entry in pair_manifest)
    }
    selected_probability_format_counts = {
        probability_format: sum(1 for entry in pair_manifest if entry.get("probability_format") == probability_format)
        for probability_format in PROBABILITY_FORMATS
        if any(entry.get("probability_format") == probability_format for entry in pair_manifest)
    }

    eval_cfg = {
        "task": "reward_model_pairwise_preference_eval",
        "base_model": args.base_model,
        "model_path": args.model_path,
        "tokenizer_name": args.tokenizer_name,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
        "format_mode": args.format_mode,
        "system_prompt": args.system_prompt,
        "system_prompt_source": getattr(args, "system_prompt_source", None),
        "prompt_suffix": args.prompt_suffix,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "reward_output_index": args.reward_output_index,
        "tie_epsilon": args.tie_epsilon,
        "num_pairs": target_total,
        "num_pairs_completed": target_completed,
        "start_position": args.start_position,
        "end_position": args.end_position,
        "stop_after": args.stop_after,
        "dataset": args.dataset,
        "custom_csv": args.custom_csv,
        "csv_path": args.csv_path,
        "save_every": args.save_every,
        "backup_every": args.backup_every,
        "selection_unit": "pair_rows",
        "raw_pair_rows_in_slice": (selection_stats or {}).get("raw_pair_rows_in_slice", target_total),
        "exact_duplicate_rows_skipped": (selection_stats or {}).get("exact_duplicate_rows_skipped", 0),
        "selected_pair_ids": target_pair_ids,
        "selected_subset_type_counts": selected_subset_type_counts,
        "selected_probability_format_counts": selected_probability_format_counts,
        "selected_pairs": pair_manifest,
    }

    stored_results = [project_result_row_for_output(row, include_responses=True) for row in ordered_results]
    output_data = convert_numpy(
        {
            "evaluation_config": eval_cfg,
            "metrics": summary_payload["metrics"],
            "num_total": summary_payload["num_total"],
            "num_correct": summary_payload["num_correct"],
            "num_incorrect": summary_payload["num_incorrect"],
            "num_ties": summary_payload["num_ties"],
            "num_truncated_pairs": summary_payload["num_truncated_pairs"],
            "metrics_by_subset_type": summarize_results_by_field(
                ordered_results, pair_manifest, "subset_type", ordered_values=list(SUBSET_TYPES)
            ),
            "metrics_by_probability_format": summarize_results_by_field(
                ordered_results, pair_manifest, "probability_format", ordered_values=list(PROBABILITY_FORMATS)
            ),
            "results": stored_results,
            "resume_records": compact_results_for_resume(ordered_results),
            "progress": {
                "target_total": target_total,
                "completed": target_completed,
                "remaining": max(target_total - target_completed, 0),
                "next_pair_id": next_pair_id,
                "checkpoint_index": target_completed,
            },
            "progress_by_subset_type": summarize_progress_by_subset_type(ordered_results, pair_manifest),
        }
    )

    atomic_write_json(output_path, output_data)
    if create_backup:
        shutil.copy2(output_path, f"{output_path}.bak")


def auto_output_path(args) -> str:
    base = (args.model_path or args.base_model or "reward_model").rstrip("/").split("/")[-1]
    dataset = args.dataset if args.custom_csv is None else Path(args.custom_csv).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reward_eval_{base}_{dataset}_{timestamp}.json"
    return resolve_path(os.path.join("results", filename))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default=None, help="Reward model ID or local path")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional PEFT adapter path. If --base_model is omitted, this is treated as the full reward-model path.",
    )
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Optional tokenizer override")
    parser.add_argument(
        "--dataset",
        type=str,
        default="reward_model_validation",
        choices=list(DATASET_ALIASES.keys()),
        help="Built-in reward-model dataset alias (ignored if --custom_csv is provided)",
    )
    parser.add_argument(
        "--custom_csv",
        type=str,
        default=None,
        help="Advanced: path to custom pairwise CSV dataset (overrides --dataset)",
    )
    parser.add_argument("--list_datasets", action="store_true", help="List built-in reward-model datasets and exit")
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=None,
        help="Number of pair rows to evaluate after deduping exact duplicate prompt/chosen/rejected rows (default: all selected rows)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (auto-generated if omitted)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of prompt+response pairs to score in parallel (default: 16)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum tokenized input length per prompt+response transcript (default: 4096)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help=(
            "Shared system prompt included when formatting reward-model inputs. "
            "If omitted, the repo uses the normal built-in default except for Gemma 3 12B, "
            "which now defaults to no system prompt."
        ),
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Optional extra instruction appended to each cleaned prompt before scoring",
    )
    parser.add_argument(
        "--format_mode",
        choices=["auto", "chat_template", "plain_text"],
        default="auto",
        help="How to format prompt/response examples for the reward model (default: auto)",
    )
    parser.add_argument(
        "--disable_length_sort",
        action="store_true",
        help="Disable batching pending pairs by approximate length before scoring",
    )
    parser.add_argument(
        "--reward_output_index",
        type=int,
        default=None,
        help="If the reward model emits multiple logits, select this column as the scalar reward",
    )
    parser.add_argument(
        "--tie_epsilon",
        type=float,
        default=1e-6,
        help="Scores within +/- epsilon are treated as ties (default: 1e-6)",
    )
    parser.add_argument("--start_position", type=int, default=1, help="1-based dataset row position to start from")
    parser.add_argument("--end_position", type=int, default=None, help="1-based inclusive end position in dataset order")
    parser.add_argument(
        "--stop_after",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSON if present")
    parser.add_argument(
        "--save_every",
        type=int,
        default=16,
        help="Write checkpoint every N newly evaluated pair rows (default: 16, aligned with default batch_size)",
    )
    parser.add_argument(
        "--backup_every",
        type=int,
        default=80,
        help="Write .bak backup every N newly evaluated pair rows (default: 80, 0 disables backups)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Torch dtype for reward model loading: auto, float16/fp16, bfloat16/bf16, float32/fp32",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='Transformers device_map for reward model loading (default: "auto")',
    )
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow custom model/tokenizer code from the Hub")
    parser.add_argument("--no_fast_tokenizer", action="store_true", help="Disable fast tokenizer loading")
    parser.add_argument(
        "--reward_head_path",
        type=str,
        default=None,
        help=(
            "Path to reward_head.pt produced by rft_pipeline.py. Triggers the "
            "AutoModel + LoRA + separate reward-head loader instead of "
            "AutoModelForSequenceClassification. If omitted, auto-detects "
            "reward_head.pt inside --model_path. Pass '' to force the default "
            "sequence-classification loader even when one is present."
        ),
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("Built-in reward-model datasets (recommended current defaults):")
        for name, rel_path in CANONICAL_DATASET_ALIASES.items():
            resolved = resolve_path(rel_path)
            row_count = count_dataset_rows(resolved)
            row_count_text = f" ({row_count} rows)" if row_count is not None else ""
            print(f"  {name:48} -> {resolved}{row_count_text}")
        print("\nAdditional current aliases:")
        for name, rel_path in CURRENT_EXTRA_DATASET_ALIASES.items():
            resolved = resolve_path(rel_path)
            row_count = count_dataset_rows(resolved)
            row_count_text = f" ({row_count} rows)" if row_count is not None else ""
            print(f"  {name:48} -> {resolved}{row_count_text}")
        return

    if args.custom_csv:
        if args.dataset != "reward_model_validation":
            print("Note: --custom_csv overrides --dataset; using custom reward-model dataset path.")
        args.dataset = "custom"
        args.custom_csv = resolve_path(args.custom_csv)
        args.csv_path = args.custom_csv
    else:
        args.csv_path = resolve_path(DATASET_ALIASES[args.dataset])

    args.system_prompt, args.system_prompt_source = resolve_system_prompt(
        dataset_base_alias=args.dataset,
        base_model=args.base_model,
        model_path=args.model_path,
        explicit_system_prompt=args.system_prompt,
    )
    if args.system_prompt_source == DATASET_DEFAULT_SYSTEM_PROMPT_SOURCE and args.dataset != "custom":
        print(f"Using default system prompt for reward-model dataset family: {args.dataset}")
    elif args.system_prompt_source == MODEL_DEFAULT_NO_SYSTEM_PROMPT_SOURCE:
        print("Using model-specific no-system-prompt default for Gemma 3 12B reward-model formatting.")
    elif (
        args.system_prompt_source == CLI_SYSTEM_PROMPT_SOURCE
        and args.system_prompt.strip()
        and (model_uses_no_system_prompt(args.base_model) or model_uses_no_system_prompt(args.model_path))
    ):
        print(
            "WARNING: Gemma 3 12B reward-model runs in this repo normally use no system prompt. "
            "You overrode that with --system_prompt."
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
    if args.num_pairs is not None and args.num_pairs < 1:
        raise ValueError("--num_pairs must be >= 1")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("--stop_after must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.max_length < 1:
        raise ValueError("--max_length must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")
    if args.backup_every < 0:
        raise ValueError("--backup_every must be >= 0")

    output_path = resolve_path(args.output) if args.output else auto_output_path(args)
    ensure_output_path_is_safe(output_path, resume=args.resume)

    df = pd.read_csv(args.csv_path)
    df = df.loc[:, [col for col in df.columns if not str(col).startswith("Unnamed:")]].copy()
    validate_no_literal_backslash_newlines(df, args.csv_path)
    cot_summary = summarize_cot_dataframe(df)
    if cot_summary["cells_with_multiple_think_close"] > 0 or cot_summary["cells_with_extra_text_after_think"] > 0:
        print(
            "WARNING: CoT CSV has formatting issues beyond newline escapes.\n"
            f"{format_summary(Path(args.csv_path), cot_summary)}"
        )
    if cot_summary["rows_with_prompt_meta_references"] > 0:
        print(
            "WARNING: CoT CSV contains prompt-meta / instruction-referential reasoning.\n"
            f"{format_summary(Path(args.csv_path), cot_summary)}\n"
            "Audit and filter it with: "
            f"python audit_reward_model_csv.py \"{args.csv_path}\""
        )
    validate_dataset_columns(df, args.csv_path)
    all_pairs = build_preference_pairs(df)
    selected_pairs, selection_stats = select_pairs(
        all_pairs,
        start_position=args.start_position,
        end_position=args.end_position,
        num_pairs=args.num_pairs,
    )
    if not selected_pairs:
        raise ValueError("No reward-model pair rows selected after applying dataset slice arguments.")
    if args.num_pairs is None:
        print(
            "No --num_pairs provided; using the full deduplicated selected dataset slice "
            f"({len(selected_pairs)} pair rows)."
        )

    for pair in selected_pairs:
        pair["prompt"] = build_eval_prompt(pair["prompt_raw"], args.prompt_suffix)

    ordered_pair_ids = [pair["pair_id"] for pair in selected_pairs]
    results: List[Dict[str, Any]] = []

    if args.resume:
        loaded_state = load_existing_run_state(output_path, ordered_pair_ids)
        if loaded_state is not None:
            results = loaded_state["results"]
            print(f"Resuming from {loaded_state['loaded_from']}")
            if loaded_state["dropped_duplicates"]:
                print(f"Note: dropped {loaded_state['dropped_duplicates']} duplicate resumed rows by pair_id.")

    done_ids = {row.get("pair_id") for row in results if row.get("pair_id") is not None}
    pending_pairs = [pair for pair in selected_pairs if pair["pair_id"] not in done_ids]
    if args.stop_after is not None:
        pending_pairs = pending_pairs[: args.stop_after]
    if not args.disable_length_sort:
        pending_pairs = sorted(pending_pairs, key=approximate_pair_length, reverse=True)

    print("Reward model evaluation configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  CSV path: {args.csv_path}")
    print(
        "  Selected pair rows in slice: "
        f"{len(selected_pairs)} (from {selection_stats['raw_pair_rows_in_slice']} pair rows)"
    )
    if selection_stats["exact_duplicate_rows_skipped"]:
        print(
            "  Exact duplicate pair rows skipped: "
            f"{selection_stats['exact_duplicate_rows_skipped']}"
        )
    print(f"  Pending pair rows this invocation: {len(pending_pairs)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Length-aware batching: {'OFF' if args.disable_length_sort else 'ON'}")
    print(f"  Max input length: {args.max_length}")
    if args.system_prompt:
        print(
            f"  System prompt: YES ({len(args.system_prompt)} chars; "
            f"source: {getattr(args, 'system_prompt_source', 'unknown')})"
        )
    else:
        print(f"  System prompt: NO (source: {getattr(args, 'system_prompt_source', 'unknown')})")
    print(f"  Output JSON: {output_path}")
    if args.save_every % args.batch_size != 0:
        print(
            f"Note: --save_every {args.save_every} is not a multiple of --batch_size {args.batch_size}; "
            "checkpoints still happen only after finished batches."
        )
    if args.backup_every > 0 and args.backup_every % args.batch_size != 0:
        print(
            f"Note: --backup_every {args.backup_every} is not a multiple of --batch_size {args.batch_size}; "
            "backups still happen only after finished batches."
        )

    if not pending_pairs:
        print("No pending reward-model pair rows left to evaluate in this invocation. Refreshing summary JSON.")
        save_incremental(
            output_path,
            args,
            results,
            selected_pairs,
            selection_stats=selection_stats,
            create_backup=False,
        )
        return

    model, tokenizer = load_reward_model(args)

    session_evaluated = 0
    while pending_pairs:
        batch = pending_pairs[: args.batch_size]
        pending_pairs = pending_pairs[args.batch_size :]
        try:
            batch_results = score_batch(
                model=model,
                tokenizer=tokenizer,
                batch_pairs=batch,
                system_prompt=args.system_prompt,
                format_mode=args.format_mode,
                max_length=args.max_length,
                reward_output_index=args.reward_output_index,
            )
        except RuntimeError as exc:
            if is_batch_size_related_runtime_error(exc):
                print_batch_size_troubleshooting_hint(args.batch_size)
            raise

        for row in batch_results:
            row["predicted_preference"] = predict_preference(
                row["accepted_score"],
                row["rejected_score"],
                args.tie_epsilon,
            )
            row["is_correct"] = row["predicted_preference"] == "accepted"

        results.extend(batch_results)
        session_evaluated += len(batch_results)

        crossed_save_boundary = args.save_every <= 1 or any(
            n % args.save_every == 0 for n in range(session_evaluated - len(batch_results) + 1, session_evaluated + 1)
        )
        crossed_backup_boundary = args.backup_every > 0 and any(
            n % args.backup_every == 0 for n in range(session_evaluated - len(batch_results) + 1, session_evaluated + 1)
        )
        if crossed_save_boundary or crossed_backup_boundary or not pending_pairs:
            save_incremental(
                output_path,
                args,
                results,
                selected_pairs,
                selection_stats=selection_stats,
                create_backup=crossed_backup_boundary,
            )

    final_summary = summarize_pairwise_results(dedupe_results_by_pair_id(results, ordered_pair_ids))
    print("Finished reward-model evaluation slice.")
    print(f"  Pairwise accuracy: {final_summary['metrics']['pairwise_accuracy']:.4f}")
    print(f"  Pairwise accuracy (ties=0.5): {final_summary['metrics']['pairwise_accuracy_ties_half_credit']:.4f}")
    print(f"  Mean score margin: {final_summary['metrics']['mean_score_margin']:.4f}")
    print(f"  Preference log loss: {final_summary['metrics']['preference_log_loss']:.4f}")


if __name__ == "__main__":
    main()
