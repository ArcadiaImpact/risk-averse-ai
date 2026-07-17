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

from utils.answer_parser import infer_option_label_style, parse_choice_with_strategy
from utils.dataset_schema_utils import ensure_option_level_dataframe
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



# ---------------------------------------------------------------------------
# Library split: dataset loading, scoring, and the shared message builder moved
# out of this monolith into focused modules. evaluate.py keeps the CLI, the
# local GPU backends (vllm parity anchor + transformers + steering), and the
# incremental save/resume IO; it re-exports the moved names so existing
# `from evaluate import ...` callers keep working. The library-first async API
# (EvalConfig + run_evaluation) is re-exported at the bottom of this file.
from situations import (
    CANONICAL_DATASET_ALIASES,
    CURRENT_EXTRA_DATASET_ALIASES,
    DATASET_ALIASES,
    DATASET_VARIANT_PATHS,
    DATASET_VARIANT_SYNONYMS,
    DEFAULT_EVAL_TEMPERATURE,
    EXTRA_DATASET_ALIASES,
    PROBABILITY_FORMATS,
    SOURCE_STAKES,
    SUBSET_TYPES,
    annotate_rows_with_situation_metadata,
    build_eval_prompt,
    build_situation_manifest,
    build_situation_manifest_index,
    build_situations,
    filter_lin_only_situations,
    label_to_option_number,
    normalize_dataset_variant,
    resolve_builtin_dataset_path,
    resolve_default_num_situations,
    resolve_path,
    validate_dataset_columns,
)
from utils.scoring import (
    count_parse_failures,
    format_pct_metric,
    project_failed_response_for_output,
    project_result_row_for_output,
    summarize_manifest_counts,
    summarize_progress_by_field,
    summarize_result_payload,
    summarize_results,
    summarize_results_by_field,
)
from generation import build_messages
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


def generate_response_openai(
    *,
    eval_prompts: List[str],
    system_prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    max_new_tokens: int,
    disable_thinking: bool,
    base_url: str,
    api_key: str,
    endpoint_model: str,
    endpoint_renderer: Optional[str] = None,
):
    """Generate responses against an OpenAI-compatible endpoint (the Tinker shim).

    The endpoint renders each request server-side with the configured
    tinker-cookbook renderer, so chat templating happens in the shim, not here.
    top_k and seed ride in ``extra_body`` (the shim forwards them to Tinker's
    SamplingParams); ``renderer`` selects thinking-enabled vs disable-thinking.
    Requests are issued concurrently across the batch via threads, mirroring the
    vLLM backend's per-batch parallelism.
    """
    from concurrent.futures import ThreadPoolExecutor

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai backend requested, but the `openai` client is not installed. "
            "Install the serve extra: pip install 'risk-averse-ai[serve]'."
        ) from exc

    client = OpenAI(base_url=base_url, api_key=api_key)
    extra_body: Dict[str, Any] = {"top_k": top_k, "seed": seed}
    if endpoint_renderer:
        extra_body["renderer"] = endpoint_renderer

    def _one(eval_prompt: str):
        messages = build_messages(eval_prompt, system_prompt)
        resp = client.chat.completions.create(
            model=endpoint_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        n_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return text, (n_tokens or 0), choice.finish_reason

    gen_start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, len(eval_prompts))) as ex:
        out = list(ex.map(_one, eval_prompts))
    gen_elapsed = time.time() - gen_start

    responses = [o[0] for o in out]
    generated_token_counts = [o[1] for o in out]
    metadata = [{"finish_reason": o[2], "stop_reason": None} for o in out]
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
    openai_base_url: Optional[str] = None,
    openai_api_key: str = "EMPTY",
    openai_endpoint_model: Optional[str] = None,
    openai_endpoint_renderer: Optional[str] = None,
):
    """Dispatch generation to the selected inference backend."""
    if backend == "openai":
        return generate_response_openai(
            eval_prompts=eval_prompts,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_new_tokens=max_new_tokens,
            disable_thinking=disable_thinking,
            base_url=openai_base_url,
            api_key=openai_api_key,
            endpoint_model=openai_endpoint_model,
            endpoint_renderer=openai_endpoint_renderer,
        )
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
            openai_base_url=getattr(args, "base_url", None),
            openai_api_key=getattr(args, "api_key", "EMPTY"),
            openai_endpoint_model=getattr(args, "endpoint_model", None) or args.base_model,
            openai_endpoint_renderer=getattr(args, "endpoint_renderer", None),
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


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluate.py argument parser. Shared by the CLI (`main`) and by
    `run_evaluation_from_config`, so the flow's in-process caller gets the exact
    same defaults as the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm", "openai"],
        default="vllm",
        help=(
            "Inference backend to use (default: vllm). 'openai' talks to an "
            "OpenAI-compatible endpoint (e.g. the local Tinker shim) via "
            "--base_url; no GPU is used locally."
        ),
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="OpenAI-compatible endpoint base URL (required for --backend openai, e.g. http://127.0.0.1:8100/v1)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the OpenAI-compatible endpoint (default: EMPTY; the local shim ignores it)",
    )
    parser.add_argument(
        "--endpoint_model",
        type=str,
        default=None,
        help=(
            "Model string sent to the OpenAI-compatible endpoint per request "
            "(the shim uses it to select the arm: a base model name or a "
            "tinker://.../sampler_weights/... checkpoint path). Defaults to "
            "--base_model when omitted."
        ),
    )
    parser.add_argument(
        "--endpoint_renderer",
        type=str,
        default=None,
        help=(
            "Renderer name the shim should use for this run (thinking-enabled "
            "for risk datasets, disable-thinking for MMLU). Omit to use the "
            "shim's server default."
        ),
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

    return parser


def run_cli_evaluation(args):
    """Run one local-backend / steering evaluation from a CLI `args` namespace.

    `args` is a fully-populated namespace (all `build_parser` defaults resolved).
    Returns the single-alpha summary dict (the common case: metrics, num_valid,
    num_parse_failed, num_total, ...) or, for a multi-alpha sweep, the sweep
    payload. This drives the local vLLM/transformers backends and steering alpha
    sweeps; the library-first async ``run_evaluation`` (runner.py) is the primary
    API for the client (openai) path.
    """
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
    nonzero_alphas_early = any(abs(a) > 0 for a in alphas)
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
    elif args.backend == "openai":
        # No local model: generation goes to the OpenAI-compatible endpoint
        # (the Tinker shim). Templating and sampling happen server-side.
        if not args.base_url:
            raise ValueError(
                "--backend openai requires --base_url (e.g. http://127.0.0.1:8100/v1)."
            )
        if args.steering_direction_path or nonzero_alphas_early:
            raise ValueError("Steering is not supported with the openai endpoint backend.")
        print(
            f"Using OpenAI-compatible endpoint: {args.base_url} "
            f"(model={args.endpoint_model or args.base_model}, "
            f"renderer={args.endpoint_renderer or 'server-default'})"
        )
        model = None
        tokenizer = None
        lora_request = None
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
    elif args.backend == "transformers":
        layers = get_decoder_layers(model)
        n_layers = len(layers)
    # openai backend: no local model, no steering (guarded above).

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

        result = sweep_payload
    else:
        result = per_alpha_summaries[0]

    del model
    gc.collect()
    if args.backend == "transformers" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_evaluation_from_config(**overrides):
    """In-process entrypoint: build args with `build_parser` defaults, apply the
    given overrides, and run. The flow calls this instead of shelling out, so the
    system prompt and generation params pass as plain Python arguments end to end.

    Keyword names match the CLI flags without the leading dashes (e.g.
    ``base_url=..., dataset=..., num_situations=..., system_prompt=...``).
    """
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"run_evaluation_from_config got unknown option {key!r}")
        setattr(args, key, value)
    return run_cli_evaluation(args)


def _eval_config_from_args(args) -> "EvalConfig":
    """Build a library ``EvalConfig`` from a CLI `args` namespace (openai path)."""
    return EvalConfig(
        dataset=args.dataset,
        num_situations=args.num_situations,
        base_model=args.base_model,
        backend="openai",
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        reasoning_max_tokens=args.reasoning_max_tokens,
        system_prompt=args.system_prompt,
        prompt_suffix=args.prompt_suffix,
        force_default_system_prompt=args.force_default_system_prompt,
        dataset_variant=args.dataset_variant,
        custom_csv=args.custom_csv,
        lin_only=args.lin_only,
        start_position=args.start_position,
        end_position=args.end_position,
        output=args.output,
        save_responses=not args.no_save_responses,
    )


def main():
    args = build_parser().parse_args()

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

    if args.backend == "openai":
        # URLs are a CLI concern, never a library one: the flag builds a client
        # here and the library only ever sees the client object. --base_url
        # points at a running shim (out-of-process face); omit it to sample
        # in-process via Tinker directly.
        import asyncio

        # serving lives one dir up under src/; put src on the path so the CLI
        # can build a client without being installed as a package.
        if str(EVAL_DIR.parent) not in sys.path:
            sys.path.insert(0, str(EVAL_DIR.parent))
        from serving import client as make_client

        cfg = _eval_config_from_args(args)
        c = make_client(
            model=args.endpoint_model or args.base_model,
            renderer=args.endpoint_renderer,
            base_url=args.base_url or None,
            api_key=args.api_key,
        )
        asyncio.run(run_evaluation(cfg, c))
        return

    run_cli_evaluation(args)


# Library-first API, re-exported so `from evaluate import run_evaluation,
# EvalConfig` works. `run_evaluation` here is the async runner (runner.py); the
# CLI's local-backend path is `run_cli_evaluation` above. Imported at the bottom
# so evaluate is fully defined first (runner imports evaluate lazily for the
# local backends, so top-level order matters).
from config import EvalConfig, EvalResult  # noqa: E402
from runner import run_evaluation  # noqa: E402


if __name__ == "__main__":
    main()
