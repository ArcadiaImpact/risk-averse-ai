from __future__ import annotations

import inspect
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(f"Expected a JSON list, JSON object with rows, or JSONL file: {source}")


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp.replace(output)


def maybe_limit_rows(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples and max_samples > 0:
        return rows[:max_samples]
    return rows


def resolve_dtype(dtype_name: str, model_name: str = "") -> torch.dtype:
    key = dtype_name.lower()
    if key == "auto":
        # Prefer bf16 on GPUs that support it. It avoids fp16 gradient scaling
        # failures and has been more stable for QLoRA/TRL across model families.
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def trainer_precision_flags(torch_dtype: torch.dtype) -> dict[str, bool]:
    """Return mutually exclusive precision flags for Transformers/TRL trainers."""
    return {
        "bf16": torch_dtype == torch.bfloat16,
        "fp16": torch_dtype == torch.float16,
    }


def filter_kwargs_for_dataclass(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop config options not supported by the installed TRL version."""
    params = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def first_supported_dataclass_kwarg(cls, *names: str) -> str | None:
    """Return the first supported kwarg name for a dataclass-like config."""
    params = inspect.signature(cls.__init__).parameters
    for name in names:
        if name in params:
            return name
    return None


def setup_hf_cache(cache_dir: str | Path | None) -> None:
    if not cache_dir:
        return
    cache_path = Path(cache_dir)
    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(cache_path / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path / "transformers")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def record_stage(stage_name: str, started: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "stage": stage_name,
        "finished_at_utc": utc_now_iso(),
        "runtime_seconds": time.time() - started,
    }
    if extra:
        payload.update(extra)
    return payload


# --- Hugging Face Hub adapter upload --------------------------------------
# Per the ML experiments runbook: HF Hub = full archive, local/volume = mirror. Adapters
# from real training runs should be pushed to the Hub immediately after training so they
# survive volume corruption/cleanup. These helpers are shared by the SFT and DPO trainers.

# A run is treated as a throwaway "smoke" (no HF nag) when it trains on very few examples.
SMOKE_MAX_SAMPLES_THRESHOLD = 256


def warn_if_hub_upload_disabled(
    *,
    push_to_hub: bool,
    hub_repo_id: str,
    max_train_samples: int,
    dataset_size: int,
) -> None:
    """Print a loud banner when a real (non-smoke) run is NOT uploading its adapter to HF.

    Real-run heuristic: no per-run sample cap (max_train_samples == 0) or a cap above the
    smoke threshold. Smokes stay quiet.
    """
    is_real_run = (max_train_samples == 0) or (max_train_samples > SMOKE_MAX_SAMPLES_THRESHOLD)
    if push_to_hub and hub_repo_id:
        return
    if not is_real_run:
        print("[hub] note: HF upload off (small/smoke run); skipping the archival nag.", flush=True)
        return
    reason = "push_to_hub is OFF" if not push_to_hub else "no --hub_repo_id given"
    banner = "!" * 78
    print(
        f"\n{banner}\n"
        f"[hub] WARNING: this looks like a REAL run ({dataset_size} examples) but {reason}.\n"
        f"[hub] The ML runbook says HF Hub = full archive; the local/volume copy is only a\n"
        f"[hub] mirror and is not durable. Pass --push_to_hub --hub_repo_id <org/repo> so the\n"
        f"[hub] final adapter is uploaded, or accept that this adapter may be lost.\n"
        f"{banner}\n",
        flush=True,
    )


def push_adapter_to_hub(
    *,
    push_to_hub: bool,
    hub_repo_id: str,
    final_adapter_dir: str | Path,
    run_name: str,
    path_in_repo: str = "",
    private: bool = True,
) -> str | None:
    """Upload final_adapter/ to HF Hub. Returns the repo URL, or None if skipped/failed.

    Token is read from the HF_TOKEN environment variable (set via a Modal secret on the
    remote machine), never passed on the command line. Each run goes in its own subfolder
    (path_in_repo, defaulting to run_name) so one repo can hold a whole sweep.
    """
    if not push_to_hub:
        return None
    if not hub_repo_id:
        print("[hub] --push_to_hub set but hub_repo_id empty; skipping upload.", flush=True)
        return None
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[hub] HF_TOKEN not in env; skipping upload.", flush=True)
        return None
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # pragma: no cover
        print(f"[hub] huggingface_hub not importable: {exc}; skipping upload.", flush=True)
        return None
    sub = path_in_repo or run_name
    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=hub_repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=hub_repo_id,
            repo_type="model",
            folder_path=str(final_adapter_dir),
            path_in_repo=sub,
            commit_message=f"Add adapter for {run_name}",
        )
        url = f"https://huggingface.co/{hub_repo_id}/tree/main/{sub}"
        print(f"[hub] uploaded final_adapter -> {url}", flush=True)
        return url
    except Exception as exc:  # pragma: no cover
        print(f"[hub] upload FAILED: {exc}", flush=True)
        return None


def _checkpoint_looks_complete(path: Path) -> bool:
    """A TRL/PEFT checkpoint dir is considered complete if it contains a real
    weights file (adapter_model.safetensors for LoRA, or model.safetensors / pytorch_model.bin
    for full models) AND a trainer_state.json. We deliberately do not trust dirs that have
    only README.md — TRL writes the auto-generated model card before the weights, so a crash
    mid-save leaves exactly that broken state."""
    if not path.is_dir():
        return False
    if not (path / "trainer_state.json").exists():
        return False
    if (path / "adapter_model.safetensors").exists():
        return True
    if (path / "model.safetensors").exists():
        return True
    if (path / "pytorch_model.bin").exists():
        return True
    return False


def clean_partial_checkpoints(checkpoints_dir: str | Path) -> list[str]:
    """Delete checkpoint-* directories that are incomplete (missing weights or trainer_state).
    Returns the list of paths that were removed. Safe to call at training startup so that
    auto-resume never tries to load a half-written checkpoint left by a previous crash."""
    import shutil

    root = Path(checkpoints_dir)
    if not root.exists():
        return []
    removed: list[str] = []
    for path in sorted(root.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        if not re.fullmatch(r"checkpoint-\d+", path.name):
            continue
        if _checkpoint_looks_complete(path):
            continue
        shutil.rmtree(path)
        removed.append(str(path))
    return removed


def latest_checkpoint(checkpoints_dir: str | Path, require_complete: bool = True) -> str | None:
    """Return the highest-numbered checkpoint-N directory. If require_complete=True (the default),
    skip any directory that does not contain real weights + trainer_state.json. Set require_complete=False
    to fall back to the old permissive behaviour."""
    root = Path(checkpoints_dir)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match:
            continue
        if require_complete and not _checkpoint_looks_complete(path):
            continue
        candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return str(max(candidates, key=lambda item: item[0])[1])


def resolve_resume_checkpoint(run_dir: str | Path, checkpoints_dir: str | Path, resume_arg: str) -> str | None:
    """Resolve `--resume_from_checkpoint`. Accepted values:
      * ""          : AUTO. Return the latest complete checkpoint if any exists, else None.
                      This is what you want under Modal/Lambda where workers can be preempted
                      and the function input gets re-scheduled. Starting from scratch in that
                      case throws away hours of compute.
      * "none"      : Force a fresh start. Useful when configs/data have changed and old
                      checkpoints should NOT be used.
      * "latest"    : Same as "" but error if no complete checkpoint exists.
      * "<path>"    : Resume from the explicit path (absolute, or relative to run/checkpoints dir).
    """
    if resume_arg == "none":
        return None
    if not resume_arg:
        checkpoint = latest_checkpoint(checkpoints_dir, require_complete=True)
        return checkpoint  # may be None on a first launch
    if resume_arg == "latest":
        checkpoint = latest_checkpoint(checkpoints_dir, require_complete=True)
        if checkpoint is None:
            raise FileNotFoundError(f"Asked to resume from latest checkpoint, but no complete checkpoints exist in {checkpoints_dir}")
        return checkpoint
    checkpoint = Path(resume_arg)
    if not checkpoint.is_absolute():
        run_relative = Path(run_dir) / resume_arg
        checkpoints_relative = Path(checkpoints_dir) / resume_arg
        checkpoint = run_relative if run_relative.exists() else checkpoints_relative
    if not checkpoint.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def normalize_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def regex_reward(completions: list[Any], target_regex: list[str] | str | None = None, **_: Any) -> list[float]:
    if target_regex is None:
        return [0.0 for _ in completions]
    targets = target_regex if isinstance(target_regex, list) else [target_regex] * len(completions)
    rewards: list[float] = []
    for completion, pattern in zip(completions, targets):
        text = normalize_completion_text(completion).strip().lower()
        rewards.append(1.0 if re.search(pattern, text, flags=re.IGNORECASE) else 0.0)
    return rewards


def messages_from_prompt(prompt: Any, system_prompt: str = "") -> list[dict[str, str]]:
    if isinstance(prompt, list):
        return prompt
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": str(prompt)})
    return messages


def sft_text_from_row(row: dict[str, Any], tokenizer, system_prompt: str = "") -> str:
    messages = row.get("messages")
    if messages is None:
        prompt = row.get("prompt", "")
        response = row.get("response", row.get("completion", ""))
        messages = messages_from_prompt(prompt, system_prompt=system_prompt)
        messages.append({"role": "assistant", "content": str(response)})
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)


def rl_prompt_from_row(row: dict[str, Any], system_prompt: str = "") -> str | list[dict[str, str]]:
    if "messages" in row:
        messages = list(row["messages"])
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return messages
    return messages_from_prompt(row.get("prompt", ""), system_prompt=system_prompt)


def patch_gemma3_token_type_ids(model):
    config = getattr(model, "config", None)
    if getattr(config, "model_type", "") != "gemma3":
        return model
    if getattr(model, "_fast_skeleton_gemma3_token_type_ids_patch", False):
        return model

    original_forward = model.forward

    def squeeze_extra_singleton_axis(value):
        if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[1] == 1:
            return value.squeeze(1)
        return value

    def forward_with_text_token_type_ids(*args, **kwargs):
        args = tuple(squeeze_extra_singleton_axis(value) for value in args)
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key in kwargs:
                kwargs[key] = squeeze_extra_singleton_axis(kwargs[key])
        if kwargs.get("token_type_ids") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is not None:
                kwargs["token_type_ids"] = torch.zeros_like(input_ids)
        output = original_forward(*args, **kwargs)
        logits = getattr(output, "logits", None)
        if isinstance(logits, torch.Tensor) and logits.ndim == 4 and logits.shape[1] == 1:
            output.logits = logits.squeeze(1)
        return output

    model.forward = forward_with_text_token_type_ids
    model._fast_skeleton_gemma3_token_type_ids_patch = True
    return model
