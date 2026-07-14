"""Generic QLoRA DPO trainer.

Pairs with `train_sft_lora.py`. Same project-agnostic conventions:
  - 4-bit QLoRA, LoRA r=32 alpha=64 default.
  - Input is JSONL with `{prompt, chosen, rejected}` (and arbitrary extra fields, ignored here).
  - Chat template is applied via `tokenizer.apply_chat_template`.
  - Qwen3 training keeps thinking enabled by default; model families that normally use no
    system prompt default to an empty system prompt.
  - Resume from latest checkpoint via `--resume_from_checkpoint latest` (or empty for auto).
  - Optional warm-start from a previous adapter via `--init_adapter_path`. The adapter is
    loaded BEFORE LoRA injection: we merge it into the base model first, then add a fresh
    LoRA on top. This is how you continue training on more data from a prior run without
    duplicating the LoRA stack.
  - Atomic final adapter save (write to .tmp then rename) so a crash mid-save doesn't leave
    a broken final_adapter/ directory.

Run config and summary mirror train_sft_lora.py so downstream tooling can treat the run dirs
interchangeably.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    set_seed,
)
from trl import DPOConfig, DPOTrainer
from risk_averse_prompts import DEFAULT_SYSTEM_PROMPT, model_uses_no_system_prompt


class SaveAtStepsCallback(TrainerCallback):
    """Force a checkpoint save at an explicit set of optimizer steps.

    Used to capture a custom (e.g. log-spaced) trajectory of checkpoints, rather than the
    uniform `save_steps` interval. The regular interval is disabled by setting save_steps to
    a value larger than the run, so saves happen ONLY at these steps (plus the final adapter).
    """

    def __init__(self, target_steps):
        self.target_steps = set(int(s) for s in target_steps)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.target_steps:
            control.should_save = True
        return control

from fast_llm_common import (
    clean_partial_checkpoints,
    filter_kwargs_for_dataclass,
    maybe_limit_rows,
    patch_gemma3_token_type_ids,
    push_adapter_to_hub,
    read_rows,
    record_stage,
    resolve_dtype,
    resolve_resume_checkpoint,
    setup_hf_cache,
    trainer_precision_flags,
    utc_now_iso,
    warn_if_hub_upload_disabled,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic QLoRA DPO trainer.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--train_path", required=True,
                        help="JSONL with {prompt, chosen, rejected} per row.")
    parser.add_argument("--output_root", default="runs")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for DPO training.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO KL strength.")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Max combined prompt+completion length (DPO).")
    parser.add_argument("--max_prompt_length", type=int, default=1792)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_at_pairs", default="",
                        help="Comma-separated list of training-example counts (pairs) to save a "
                             "checkpoint at, e.g. '100,250,500,1000,2000,4000,8000,16000'. "
                             "Each is converted to an optimizer step via the effective batch size "
                             "(per_device_train_batch_size * gradient_accumulation_steps). When set, "
                             "the uniform save_steps interval is disabled and checkpoints are saved "
                             "ONLY at these points (plus the final adapter). Use for log-spaced "
                             "trajectory pilots.")
    parser.add_argument("--save_total_limit", type=int, default=5,
                        help="Max number of checkpoints to keep. Pass -1 to keep all.")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--hf_cache_dir", default="")
    parser.add_argument("--resume_from_checkpoint", default="")
    parser.add_argument("--init_adapter_path", default="",
                        help="If set, merge this LoRA adapter into the base model before adding "
                             "the fresh LoRA. Use this to continue training on additional data.")
    parser.add_argument("--qwen3_disable_thinking",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="For Qwen3 models, pass enable_thinking=False to the chat template.")
    # HF Hub upload of the final adapter (token read from HF_TOKEN env, never argv).
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False,
                        help="Upload final_adapter to HF Hub after training (reads HF_TOKEN env).")
    parser.add_argument("--hub_repo_id", default="",
                        help="Target HF repo for the final adapter upload.")
    parser.add_argument("--hub_path_in_repo", default="",
                        help="Subfolder in the repo for this run's adapter. Defaults to run_name.")
    args = parser.parse_args()
    if args.system_prompt == DEFAULT_SYSTEM_PROMPT and model_uses_no_system_prompt(args.model_name):
        args.system_prompt = ""
    return args


def resolve_run_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = args.model_name.split("/")[-1].replace(".", "_")
        run_name = f"{timestamp}_{model_slug}_dpo_lora"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _is_qwen3(model_name: str) -> bool:
    return "qwen3" in model_name.lower()


def _apply_chat_template_row(
    row: Dict[str, Any],
    tokenizer: Any,
    *,
    system_prompt: str,
    qwen3_disable_thinking: bool,
    model_name: str,
) -> Dict[str, str]:
    """Convert a {prompt, chosen, rejected} row to chat-templated strings for TRL DPO."""
    messages_prompt: List[Dict[str, str]] = []
    if system_prompt:
        messages_prompt.append({"role": "system", "content": system_prompt})
    messages_prompt.append({"role": "user", "content": row["prompt"]})

    extra_template_kwargs: Dict[str, Any] = {}
    if qwen3_disable_thinking and _is_qwen3(model_name):
        extra_template_kwargs["enable_thinking"] = False

    try:
        prompt_str = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True, **extra_template_kwargs
        )
    except TypeError:
        # Tokenizer doesn't accept enable_thinking — fall back without it.
        prompt_str = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )

    return {
        "prompt": prompt_str,
        "chosen": str(row["chosen"]),
        "rejected": str(row["rejected"]),
    }


def _load_base_model(
    args: argparse.Namespace,
    torch_dtype: torch.dtype,
) -> Any:
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    patch_gemma3_token_type_ids(model)

    if args.init_adapter_path:
        # Warm-start: load and merge an existing LoRA, then strip it so the next LoRA is fresh.
        adapter_path = Path(args.init_adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"init_adapter_path does not exist: {adapter_path}")
        print(f"[init_adapter] loading {adapter_path} for warm start", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path))
        try:
            model = model.merge_and_unload()
            print("[init_adapter] merged adapter weights into base model", flush=True)
        except Exception as exc:
            raise RuntimeError(
                "Could not merge_and_unload the init adapter. With 4-bit base models, merging "
                "may require re-loading in fp16/bf16. Re-run with --no-load_in_4bit if needed."
            ) from exc
    return model


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    setup_hf_cache(args.hf_cache_dir)
    run_dir = resolve_run_dir(args)
    checkpoints_dir = run_dir / "checkpoints"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # TRL DPO is happier with left padding for prompts

    rows = maybe_limit_rows(read_rows(args.train_path), args.max_train_samples)
    if not rows:
        raise RuntimeError(f"No rows loaded from {args.train_path}")

    templated = [
        _apply_chat_template_row(
            row, tokenizer,
            system_prompt=args.system_prompt,
            qwen3_disable_thinking=args.qwen3_disable_thinking,
            model_name=args.model_name,
        )
        for row in rows
    ]
    dataset = Dataset.from_list(templated)

    warn_if_hub_upload_disabled(
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        max_train_samples=args.max_train_samples,
        dataset_size=len(dataset),
    )

    torch_dtype = resolve_dtype(args.torch_dtype, args.model_name)
    model = _load_base_model(args, torch_dtype)

    # With gradient checkpointing on a (quantized) base, the input embeddings don't
    # require grad by default. That triggers torch's "None of the inputs have
    # requires_grad=True" warning and can silently zero gradients through the
    # checkpointed blocks. Registering this hook guarantees grads reach the LoRA params.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Custom log-spaced checkpoint schedule (optional). Convert each requested example-count
    # (pairs) to an optimizer step via the effective batch size, then save ONLY at those steps.
    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
    save_at_steps: List[int] = []
    if args.save_at_pairs.strip():
        pairs = [int(p) for p in args.save_at_pairs.split(",") if p.strip()]
        seen = set()
        for p in pairs:
            step = max(1, round(p / effective_batch))
            actual_pairs = step * effective_batch
            if actual_pairs != p:
                print(f"[save_at_pairs] {p} pairs not divisible by effective batch "
                      f"{effective_batch}; rounding to step {step} = {actual_pairs} pairs", flush=True)
            if step not in seen:
                seen.add(step)
                save_at_steps.append(step)
        save_at_steps.sort()
        print(f"[save_at_pairs] effective_batch={effective_batch}; "
              f"saving checkpoints at steps {save_at_steps} "
              f"(pairs {[s * effective_batch for s in save_at_steps]})", flush=True)

    # When a custom schedule is set, disable the uniform interval (huge save_steps) and keep
    # ALL checkpoints so the callback's saves are never pruned.
    effective_save_steps = 10 ** 9 if save_at_steps else args.save_steps
    save_total_limit = (
        None if (save_at_steps or args.save_total_limit <= 0) else args.save_total_limit
    )

    dpo_kwargs = filter_kwargs_for_dataclass(
        DPOConfig,
        {
            "output_dir": str(checkpoints_dir),
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "save_strategy": "steps",
            "save_steps": effective_save_steps,
            "save_total_limit": save_total_limit,
            "logging_steps": args.logging_steps,
            "report_to": args.report_to,
            **trainer_precision_flags(torch_dtype),
            "gradient_checkpointing": True,
            "remove_unused_columns": False,
        },
    )
    callbacks = [SaveAtStepsCallback(save_at_steps)] if save_at_steps else None
    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(**dpo_kwargs),
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    write_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "run_dir": str(run_dir.resolve()),
            "checkpoints_dir": str(checkpoints_dir.resolve()),
            "dataset_size": len(dataset),
            "torch_dtype_resolved": str(torch_dtype),
            "dpo_config_keys": sorted(dpo_kwargs),
        },
    )
    removed_partials = clean_partial_checkpoints(checkpoints_dir)
    if removed_partials:
        print(f"[resume] cleaned up partial checkpoint dirs: {removed_partials}", flush=True)
    resume_checkpoint = resolve_resume_checkpoint(run_dir, checkpoints_dir, args.resume_from_checkpoint)
    if resume_checkpoint:
        print(f"[resume] resuming from {resume_checkpoint}", flush=True)
    else:
        print("[resume] no checkpoint to resume from, starting fresh", flush=True)

    write_json(
        run_dir / "summary.json",
        {
            "status": "running",
            "model_name": args.model_name,
            "dataset_size": len(dataset),
            "init_adapter_path": args.init_adapter_path or None,
            "resume_from_checkpoint": resume_checkpoint,
            "started_at_utc": utc_now_iso(),
        },
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_adapter_dir = run_dir / "final_adapter"
    final_adapter_tmp = run_dir / "final_adapter.tmp"
    if final_adapter_tmp.exists():
        shutil.rmtree(final_adapter_tmp)
    trainer.save_model(str(final_adapter_tmp))
    tokenizer.save_pretrained(final_adapter_tmp)
    if final_adapter_dir.exists():
        shutil.rmtree(final_adapter_dir)
    os.rename(final_adapter_tmp, final_adapter_dir)

    hub_url = push_adapter_to_hub(
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        final_adapter_dir=final_adapter_dir,
        run_name=run_dir.name,
        path_in_repo=args.hub_path_in_repo,
    )

    write_json(
        run_dir / "summary.json",
        {
            "status": "completed",
            "model_name": args.model_name,
            "dataset_size": len(dataset),
            "init_adapter_path": args.init_adapter_path or None,
            "final_adapter_dir": str(final_adapter_dir.resolve()),
            "hub_url": hub_url,
            "timing": record_stage("dpo_train", started),
        },
    )
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
