#!/usr/bin/env python3
"""
Evaluate a (possibly LoRA-finetuned) model on MMLU-Redux using the same protocol
as the Qwen3 technical report: 5-shot, generative, exact-match accuracy.

This reproduces the lm-evaluation-harness mmlu_redux_generative task configuration
(version 4, dataset fxmarty/mmlu-redux-2.0-ok) but runs standalone so it can be
used with PEFT adapters and vLLM without needing the full harness installed.

Usage examples:

  # Base model, all 57 subjects
  python evaluate_mmlu_redux.py --model_path Qwen/Qwen3-8B

  # LoRA adapter on top of a base model
  python evaluate_mmlu_redux.py \
      --model_path /path/to/adapter \
      --base_model Qwen/Qwen3-8B

  # Use vLLM backend (much faster)
  python evaluate_mmlu_redux.py \
      --model_path Qwen/Qwen3-8B \
      --backend vllm

  # Evaluate only a subset of subjects
  python evaluate_mmlu_redux.py \
      --model_path Qwen/Qwen3-8B \
      --subjects anatomy astronomy

  # Disable thinking mode for Qwen3 models
  python evaluate_mmlu_redux.py \
      --model_path Qwen/Qwen3-8B \
      --disable_thinking

  # With steering vector (transformers OR vllm backend; vllm uses all-positions CAA)
  python evaluate_mmlu_redux.py \
      --model_path Qwen/Qwen3-8B \
      --backend vllm \
      --steering_direction_path direction.pt \
      --steering_layer 18 \
      --steering_apply_mode all_positions \
      --alphas "34"
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

sys.stdout.reconfigure(line_buffering=True)

EVAL_DIR = Path(__file__).resolve().parent
STEERING_DIR = EVAL_DIR.parent / "steering"
if str(STEERING_DIR) not in sys.path:
    sys.path.insert(0, str(STEERING_DIR))

# Re-use the steering hook from the main evaluator if available; otherwise
# define a minimal self-contained copy so this script works standalone.
try:
    from evaluate import ResidualSteeringHook, load_steering_direction
except ImportError:
    pass  # Defined below as fallback

# ── MMLU-Redux data + extraction: owned by the task dir, re-exported here ─
# The subject list, HF loader, 5-shot prompt builder, and last-letter parser now
# live in src/eval/tasks/mmlu_redux/loader.py (source of truth). Load that module
# by file path so this standalone CLI keeps working without importing the tasks
# package (and hence inspect_ai).
import importlib.util as _ilu

_LOADER_PATH = EVAL_DIR / "tasks" / "mmlu_redux" / "loader.py"
_spec = _ilu.spec_from_file_location("_mmlu_redux_loader", _LOADER_PATH)
_mmlu_loader = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mmlu_loader)

ALL_SUBJECTS = _mmlu_loader.ALL_SUBJECTS
SUBJECT_TO_CATEGORY = _mmlu_loader.SUBJECT_TO_CATEGORY
load_mmlu_redux = _mmlu_loader.load_mmlu_redux
format_question = _mmlu_loader.format_question
build_fewshot_prefix = _mmlu_loader.build_fewshot_prefix
build_prompt_text = _mmlu_loader.build_prompt_text
extract_answer = _mmlu_loader.extract_answer
build_eval_items = _mmlu_loader.build_eval_items


def build_per_question_record(
    item: Dict[str, Any],
    raw_response: str,
    save_responses: bool,
) -> Dict[str, Any]:
    """Create the saved record for a single evaluated question."""
    predicted_answer = extract_answer(raw_response)
    correct = predicted_answer is not None and predicted_answer.upper() == item["correct_answer"].upper()
    record = {
        "index": item["index"],
        "subject": item["subject"],
        "question": item["question"],
        "correct_answer": item["correct_answer"],
        "predicted_answer": predicted_answer,
        "correct": correct,
    }
    if save_responses:
        record["raw_response"] = raw_response
    return record


def summarize_per_question(
    per_question: List[Dict[str, Any]],
    subjects: List[str],
) -> Tuple[int, int, int, Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Aggregate accuracy and parse stats from saved per-question results."""
    per_subject = defaultdict(lambda: {"correct": 0, "total": 0, "parse_failures": 0})
    category_agg = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in per_question:
        subj = record["subject"]
        per_subject[subj]["total"] += 1
        if record["predicted_answer"] is None:
            per_subject[subj]["parse_failures"] += 1
        elif record["correct"]:
            per_subject[subj]["correct"] += 1

    subject_results = {}
    for subj in subjects:
        s = per_subject[subj]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        subject_results[subj] = {
            "accuracy": round(acc, 4),
            "correct": s["correct"],
            "total": s["total"],
            "parse_failures": s["parse_failures"],
        }
        cat = SUBJECT_TO_CATEGORY.get(subj, "other")
        category_agg[cat]["correct"] += s["correct"]
        category_agg[cat]["total"] += s["total"]

    category_results = {}
    for cat in ["stem", "humanities", "social_sciences", "other"]:
        a = category_agg[cat]
        acc = a["correct"] / a["total"] if a["total"] > 0 else 0.0
        category_results[cat] = {
            "accuracy": round(acc, 4),
            "correct": a["correct"],
            "total": a["total"],
        }

    total_correct = sum(s["correct"] for s in per_subject.values())
    processed_questions = len(per_question)
    total_parse_failures = sum(s["parse_failures"] for s in per_subject.values())
    return (
        total_correct,
        processed_questions,
        total_parse_failures,
        category_results,
        subject_results,
    )


def build_results_payload(
    config: Dict[str, Any],
    per_question: List[Dict[str, Any]],
    elapsed_seconds: float,
) -> Dict[str, Any]:
    """Build the saved JSON payload for either a partial or complete run."""
    subjects = config["subjects"]
    total_correct, processed_questions, total_parse_failures, category_results, subject_results = summarize_per_question(
        per_question, subjects
    )
    overall_accuracy = total_correct / processed_questions if processed_questions > 0 else 0.0
    expected_total_questions = config["expected_total_questions"]

    return {
        "config": config,
        "summary": {
            "overall_accuracy": round(overall_accuracy, 4),
            "total_correct": total_correct,
            "processed_questions": processed_questions,
            "total_questions": expected_total_questions,
            "total_parse_failures": total_parse_failures,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "is_complete": processed_questions == expected_total_questions,
        },
        "category_results": category_results,
        "subject_results": subject_results,
        "timestamp": datetime.now().isoformat(),
        "per_question": per_question,
    }


def write_results_atomic(output_path: str, payload: Dict[str, Any]) -> None:
    """Write the results JSON atomically to avoid corrupt partial files."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_suffix(output.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(output)


def validate_resume_config(existing: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Fail fast if a resume attempt changes the semantic evaluation protocol."""
    keys_to_match = [
        "model_path",
        "base_model",
        "backend",
        "num_shots",
        "system_prompt",
        "max_new_tokens",
        "disable_thinking",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "dtype",
        "subjects",
        "max_eval_examples_per_subject",
        "dataset",
        "protocol",
    ]
    for key in keys_to_match:
        if existing.get(key) != current.get(key):
            raise ValueError(
                f"Resume config mismatch for '{key}': existing={existing.get(key)!r}, current={current.get(key)!r}"
            )


# ── Steering vector support ─────────────────────────────────────────────────

if "ResidualSteeringHook" not in dir():
    class ResidualSteeringHook:
        """Minimal residual-stream steering hook (standalone fallback)."""

        def __init__(self, direction: torch.Tensor, alpha: float,
                     apply_mode: str = "last_prompt_and_current",
                     prompt_last_indices: Optional[List[int]] = None):
            self.direction = direction
            self.alpha = float(alpha)
            self.apply_mode = apply_mode
            self._handle = None

        def _broadcast(self, hidden: torch.Tensor) -> torch.Tensor:
            d = self.direction.to(device=hidden.device, dtype=hidden.dtype)
            while d.dim() < hidden.dim():
                d = d.unsqueeze(0)
            return d

        def __call__(self, module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            d = self._broadcast(hidden)
            if self.apply_mode == "all_positions":
                steered = hidden + self.alpha * d
            else:
                steered = hidden.clone()
                steered[:, -1, :] = steered[:, -1, :] + self.alpha * d
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        def register(self, model, layer_index: int):
            layers = _get_model_layers(model)
            self._handle = layers[layer_index].register_forward_hook(self, with_kwargs=False)

        def remove(self):
            if self._handle is not None:
                self._handle.remove()
                self._handle = None


if "load_steering_direction" not in dir():
    def load_steering_direction(path: str) -> torch.Tensor:
        """Load a steering vector from disk."""
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)):
            return torch.tensor(obj)
        if isinstance(obj, dict):
            for key in ("direction", "vector", "steering_direction"):
                if key in obj:
                    v = obj[key]
                    return torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        raise ValueError(f"Cannot parse steering direction from {path}")


def _get_model_layers(model):
    """Return the list of transformer layers for hook registration."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen, Llama, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2 style
    raise ValueError("Cannot find transformer layers for steering hook registration.")


# ── Model loading helpers ───────────────────────────────────────────────────

def load_model_transformers(
    model_path: str,
    base_model: Optional[str],
    disable_thinking: bool,
    dtype: str,
) -> Tuple[Any, Any]:
    """Load model and tokenizer via transformers (+ optional PEFT adapter)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32, "auto": "auto"}[dtype]

    if base_model and model_path != base_model:
        # Load base + adapter
        from peft import PeftModel
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_chat_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate_transformers(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int,
    disable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    batch_size: int,
    steering_hook: Optional["ResidualSteeringHook"] = None,
    steering_layer: Optional[int] = None,
) -> List[str]:
    """Generate responses for prompts using transformers in repeated mini-batches."""
    # Register steering hook if provided.
    # evaluate.py's ResidualSteeringHook.register(module) takes the decoder *block*,
    # not (model, layer) — resolve the block from the layer index here.
    if steering_hook is not None and steering_layer is not None:
        steering_hook.register(_get_model_layers(model)[steering_layer])

    do_sample = temperature > 0
    results = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start:start + batch_size]
            formatted = []
            for prompt in batch_prompts:
                messages = build_chat_messages(prompt, system_prompt)
                chat_kwargs = {}
                if disable_thinking:
                    chat_kwargs["enable_thinking"] = False
                formatted.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
                    )
                )

            inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generate_kwargs.update(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
            else:
                generate_kwargs.update(
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generate_kwargs)

            # `generate()` returns the full padded input plus newly generated
            # tokens. For batched decoder-only models, slice from the shared
            # padded input width, not from each row's non-pad token count.
            # Otherwise left-padded batches can leak prompt text (including
            # chat-template role markers) into the decoded "response".
            padded_input_len = inputs["input_ids"].shape[1]
            for row_idx in range(output_ids.shape[0]):
                new_tokens = output_ids[row_idx, padded_input_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append(response)
    finally:
        if steering_hook is not None:
            steering_hook.remove()

    return results


def load_vllm(
    model_path: str,
    base_model: Optional[str],
    dtype: str,
    gpu_memory_utilization: float,
    steering: bool = False,
) -> Tuple[Any, Any]:
    """Load a vLLM model and tokenizer once for repeated generation.

    When ``steering`` is True the engine is built so an all-positions CAA steering
    hook can be registered inside the vLLM worker (via
    ``vllm_steering.attach_vllm_steering``). This mirrors ``load_vllm_engine`` in
    evaluate.py:
      - ``enforce_eager=True`` so the worker's forward hook actually fires
        (CUDA-graph capture would bypass it),
      - ``enable_prefix_caching=False`` so KV is not reused across alphas/directions
        (prefix caching keys on token ids, not activations, and would silently drop
        the prompt-side steering),
      - ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` so ``apply_model`` can ship our hook
        function to the worker process.
    """
    from vllm import LLM

    load_path = model_path
    vllm_kwargs = {}

    # For PEFT adapters, we'd need to merge first or use vllm's LoRA support.
    # For simplicity, assume model_path is a merged model or a HF model ID.
    if base_model and model_path != base_model:
        print(
            "WARNING: vLLM backend with separate adapter path requires a "
            "pre-merged model. Attempting to load model_path directly. "
            "If this fails, merge the adapter first or use --backend transformers."
        )

    enable_prefix_caching = True
    if steering:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        enable_prefix_caching = False

    print(f"Loading vLLM model: {load_path}" + (" [steering: eager, no prefix cache]" if steering else ""))
    llm = LLM(
        model=load_path,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=steering,
        **vllm_kwargs,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def generate_vllm_batch(
    llm: Any,
    tokenizer: Any,
    prompts_with_chat: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int,
    disable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int,
) -> List[str]:
    """Generate one batch of responses using an already loaded vLLM model."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=seed,
        max_tokens=max_new_tokens,
        stop=["</s>"],
    )

    formatted = []
    for prompt in prompts_with_chat:
        messages = build_chat_messages(prompt, system_prompt)
        chat_kwargs = {}
        if disable_thinking:
            chat_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
        )
        formatted.append(text)

    outputs = llm.generate(formatted, sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def generate_openai_batch(
    prompts: List[str],
    system_prompt: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    base_url: str,
    api_key: str,
    endpoint_model: str,
    endpoint_renderer: Optional[str] = None,
) -> List[str]:
    """Generate one batch of responses against an OpenAI-compatible endpoint.

    The endpoint (e.g. the local Tinker shim) renders each request server-side
    with the configured renderer, so chat templating happens in the shim, not
    here. ``seed`` (and ``top_k`` when > 0) ride in ``extra_body``, which the
    shim forwards to Tinker's SamplingParams; ``renderer`` selects the
    thinking-enabled vs disable-thinking template. For MMLU we run with thinking
    disabled, so the caller passes a disable-thinking renderer and top_k -1 —
    when top_k <= 0 we omit it entirely (the shim only forwards top_k when > 0).
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
    extra_body: Dict[str, Any] = {"seed": seed}
    if top_k > 0:
        extra_body["top_k"] = top_k
    if endpoint_renderer:
        extra_body["renderer"] = endpoint_renderer

    def _one(prompt: str) -> str:
        messages = build_chat_messages(prompt, system_prompt)
        resp = client.chat.completions.create(
            model=endpoint_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""

    with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as ex:
        return list(ex.map(_one, prompts))


# ── Main evaluation loop ────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    base_model: Optional[str],
    backend: str,
    subjects: Optional[List[str]],
    num_shots: int,
    system_prompt: Optional[str],
    max_new_tokens: int,
    disable_thinking: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    seed: int,
    dtype: str,
    output_path: str,
    gpu_memory_utilization: float,
    save_responses: bool,
    max_eval_examples_per_subject: Optional[int] = None,
    batch_size: int = 64,
    save_every_batches: int = 1,
    resume: bool = False,
    checkpoint_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    steering_direction_path: Optional[str] = None,
    steering_layer: Optional[int] = None,
    steering_alpha: float = 0.0,
    steering_apply_mode: str = "last_prompt_and_current",
    base_url: Optional[str] = None,
    api_key: str = "EMPTY",
    endpoint_model: Optional[str] = None,
    endpoint_renderer: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full MMLU-Redux evaluation."""

    # vLLM steering supports only all-positions CAA (see vllm_steering.py); force it
    # so the saved config and the worker-resident hook agree.
    steering_requested = bool(steering_direction_path and steering_layer is not None)
    if backend == "vllm" and steering_requested and steering_apply_mode != "all_positions":
        print(f"NOTE: forcing steering_apply_mode -> 'all_positions' for vLLM (was {steering_apply_mode!r}).")
        steering_apply_mode = "all_positions"

    subjects = subjects or ALL_SUBJECTS
    print(f"Loading MMLU-Redux data for {len(subjects)} subjects...")
    data = load_mmlu_redux(subjects)
    eval_items = build_eval_items(subjects, data, num_shots, max_eval_examples_per_subject)
    expected_total_questions = len(eval_items)
    print(f"Total evaluation questions: {expected_total_questions}")

    config = {
        "model_path": model_path,
        "base_model": base_model,
        "backend": backend,
        "num_shots": num_shots,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "disable_thinking": disable_thinking,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "seed": seed,
        "dtype": dtype,
        "num_subjects": len(subjects),
        "subjects": subjects,
        "max_eval_examples_per_subject": max_eval_examples_per_subject,
        "batch_size": batch_size,
        "save_every_batches": save_every_batches,
        "gpu_memory_utilization": gpu_memory_utilization,
        "save_responses": save_responses,
        "dataset": "fxmarty/mmlu-redux-2.0-ok",
        "protocol": "Qwen3 tech report: 5-shot generative exact-match",
        "expected_total_questions": expected_total_questions,
        "steering_direction_path": steering_direction_path,
        "steering_layer": steering_layer,
        "steering_alpha": steering_alpha,
        "steering_apply_mode": steering_apply_mode,
        "base_url": base_url,
        "endpoint_model": endpoint_model,
        "endpoint_renderer": endpoint_renderer,
    }

    output = Path(output_path)
    per_question: List[Dict[str, Any]] = []
    previous_elapsed_seconds = 0.0
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"Output already exists: {output_path}. Pass --resume to continue from this checkpoint."
            )
        existing_payload = json.loads(output.read_text())
        validate_resume_config(existing_payload["config"], config)
        per_question = existing_payload.get("per_question", [])
        previous_elapsed_seconds = float(existing_payload.get("summary", {}).get("elapsed_seconds", 0.0))
        print(f"Resuming from {len(per_question)} completed questions in {output_path}")
        if len(per_question) >= expected_total_questions:
            print("Run is already complete; returning saved results.")
            return existing_payload
    elif resume:
        print(f"No existing output at {output_path}; starting a fresh run.")

    # Prepare steering. Transformers: a forward hook on the decoder block, built here.
    # vLLM: an all-positions CAA hook registered inside the worker AFTER the engine
    # loads (see the load section below), via vllm_steering.attach_vllm_steering.
    steering_hook = None
    if steering_requested and backend != "vllm":
        direction = load_steering_direction(steering_direction_path)
        steering_hook = ResidualSteeringHook(
            direction=direction, alpha=steering_alpha,
            apply_mode=steering_apply_mode,
        )
        print(f"Steering (transformers): layer={steering_layer}, alpha={steering_alpha}, "
              f"mode={steering_apply_mode}")

    random.seed(seed)
    if backend != "vllm":
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Generate in batches so long runs can checkpoint and resume.
    t0 = time.time()
    if backend == "vllm":
        llm, tokenizer = load_vllm(
            model_path=model_path,
            base_model=base_model,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            steering=steering_requested,
        )
        if steering_requested:
            # Register the all-positions CAA hook inside the vLLM worker, then set
            # alpha. Mirrors evaluate.py's load_vllm_engine + attach_vllm_steering.
            from vllm_steering import attach_vllm_steering
            direction = load_steering_direction(steering_direction_path)
            controller = attach_vllm_steering(
                llm, direction, steering_layer, alpha=0.0, apply_mode="all_positions",
            )
            controller.set_alpha(steering_alpha)
            print(f"Steering (vLLM): layer={steering_layer}, alpha={steering_alpha}, "
                  f"mode=all_positions, dir_norm={float(direction.norm(p=2).item()):.3f}")
    elif backend == "openai":
        # No local model: generation goes to the OpenAI-compatible endpoint (the
        # Tinker shim). Templating and sampling happen server-side.
        if not base_url:
            raise ValueError(
                "--backend openai requires --base_url (e.g. http://127.0.0.1:8100/v1)."
            )
        if steering_requested:
            raise ValueError("Steering is not supported with the openai endpoint backend.")
        model = None
        tokenizer = None
        endpoint_model = endpoint_model or base_model or model_path
        print(
            f"Using OpenAI-compatible endpoint: {base_url} "
            f"(model={endpoint_model}, renderer={endpoint_renderer or 'server-default'})"
        )
    else:
        model, tokenizer = load_model_transformers(
            model_path, base_model, disable_thinking, dtype,
        )
    processed_before = len(per_question)
    if processed_before >= expected_total_questions:
        elapsed = previous_elapsed_seconds
    else:
        for batch_num, start in enumerate(range(processed_before, expected_total_questions, batch_size), start=1):
            batch_items = eval_items[start:start + batch_size]
            batch_prompts = [item["prompt"] for item in batch_items]

            if backend == "vllm":
                responses = generate_vllm_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    prompts_with_chat=batch_prompts,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    disable_thinking=disable_thinking,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    seed=seed + start,
                )
            elif backend == "openai":
                responses = generate_openai_batch(
                    prompts=batch_prompts,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed + start,
                    base_url=base_url,
                    api_key=api_key,
                    endpoint_model=endpoint_model,
                    endpoint_renderer=endpoint_renderer,
                )
            else:
                responses = generate_transformers(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    disable_thinking=disable_thinking,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    batch_size=len(batch_prompts),
                    steering_hook=steering_hook,
                    steering_layer=steering_layer,
                )

            for item, response in zip(batch_items, responses):
                per_question.append(build_per_question_record(item, response, save_responses))

            processed_now = len(per_question)
            elapsed_so_far = previous_elapsed_seconds + (time.time() - t0)
            print(
                f"Processed {processed_now}/{expected_total_questions} questions "
                f"({processed_now - processed_before} this run)"
            )

            if batch_num % save_every_batches == 0 or processed_now == expected_total_questions:
                results = build_results_payload(config, per_question, elapsed_so_far)
                write_results_atomic(output_path, results)
                if checkpoint_callback is not None:
                    checkpoint_callback(output_path, results)
                print(f"Checkpoint saved to {output_path}")

        elapsed = previous_elapsed_seconds + (time.time() - t0)

    results = build_results_payload(config, per_question, elapsed)
    write_results_atomic(output_path, results)
    if checkpoint_callback is not None:
        checkpoint_callback(output_path, results)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"MMLU-Redux Results ({num_shots}-shot, generative)")
    print(f"Model: {model_path}")
    print(f"{'=' * 60}")
    print(
        f"Overall accuracy: {results['summary']['overall_accuracy']:.1%} "
        f"({results['summary']['total_correct']}/{results['summary']['processed_questions']})"
    )
    print(f"Parse failures: {results['summary']['total_parse_failures']}")
    print(f"\nPer-category:")
    for cat in ["stem", "humanities", "social_sciences", "other"]:
        c = results["category_results"].get(cat, {})
        print(f"  {cat:20s}: {c.get('accuracy', 0):.1%} ({c.get('correct', 0)}/{c.get('total', 0)})")
    print(f"\nPer-subject (sorted by accuracy):")
    sorted_subjs = sorted(results["subject_results"].items(), key=lambda x: x[1]["accuracy"])
    for subj, r in sorted_subjs:
        print(f"  {subj:40s}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the MMLU-Redux argument parser. Shared by the CLI (`main`) and by
    `run_mmlu_from_config`, so the flow's in-process caller gets the exact same
    defaults as the command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on MMLU-Redux (Qwen3 tech report protocol: 5-shot generative)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path", default=None,
        help="HuggingFace model ID or local path (model or adapter). Required for "
             "the transformers/vllm backends; optional for --backend openai "
             "(the endpoint selects the arm via --endpoint_model).",
    )
    parser.add_argument(
        "--base_model", default=None,
        help="Base model ID when model_path is a PEFT adapter",
    )
    parser.add_argument(
        "--backend", choices=["transformers", "vllm", "openai"], default="transformers",
        help=(
            "Inference backend (default: transformers). 'openai' talks to an "
            "OpenAI-compatible endpoint (e.g. the local Tinker shim) via "
            "--base_url; no GPU is used locally."
        ),
    )
    parser.add_argument(
        "--base_url", type=str, default=None,
        help="OpenAI-compatible endpoint base URL (required for --backend openai, e.g. http://127.0.0.1:8100/v1)",
    )
    parser.add_argument(
        "--api_key", type=str, default="EMPTY",
        help="API key for the OpenAI-compatible endpoint (default: EMPTY; the local shim ignores it)",
    )
    parser.add_argument(
        "--endpoint_model", type=str, default=None,
        help=(
            "Model string sent to the OpenAI-compatible endpoint per request (a "
            "base model name or a tinker://.../sampler_weights/... checkpoint "
            "path). Defaults to --base_model, then --model_path, when omitted."
        ),
    )
    parser.add_argument(
        "--endpoint_renderer", type=str, default=None,
        help=(
            "Renderer name the shim should use for this run (for MMLU pass a "
            "disable-thinking renderer, e.g. qwen3_disable_thinking). Omit to use "
            "the shim's server default."
        ),
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Specific subjects to evaluate (default: all 57)",
    )
    parser.add_argument(
        "--num_shots", type=int, default=5,
        help="Number of few-shot examples (default: 5, matching Qwen3 report)",
    )
    parser.add_argument(
        "--system_prompt", default=None,
        help="Optional system prompt to prepend before each MMLU question.",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Max tokens to generate per question (default: 32; only a letter is needed)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic decoding)",
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help="Top-p nucleus sampling threshold (default: 1.0)",
    )
    parser.add_argument(
        "--top_k", type=int, default=-1,
        help="Top-k sampling cutoff (default: -1, disabled)",
    )
    parser.add_argument(
        "--min_p", type=float, default=0.0,
        help="Minimum token probability cutoff (default: 0.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Random seed for sampled decoding (default: 12345)",
    )
    parser.add_argument(
        "--disable_thinking", action="store_true",
        help="Disable thinking/CoT mode for Qwen3 models",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: mmlu_redux_results_<model>_<timestamp>.json)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--no_save_responses", action="store_true",
        help="Don't save per-question responses (saves disk space)",
    )
    parser.add_argument(
        "--max_eval_examples_per_subject", type=int, default=None,
        help="Optional cap on scored examples per subject after the few-shot prefix",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Generation batch size (default: 64 for vLLM, 8 for transformers)",
    )
    parser.add_argument(
        "--save_every_batches", type=int, default=1,
        help="Write a JSON checkpoint every N generation batches (default: 1)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing output JSON with the same eval configuration",
    )

    # Steering vector options. Supported on BOTH backends: transformers applies the
    # configured apply_mode; vLLM applies all-positions CAA inside the worker.
    parser.add_argument(
        "--steering_direction_path", type=str, default=None,
        help="Path to a precomputed steering vector (.pt tensor or dict wrapper)",
    )
    parser.add_argument(
        "--steering_layer", type=int, default=None,
        help="Transformer block index (0-based) for steering injection",
    )
    parser.add_argument(
        "--alphas", type=str, default="0.0",
        help='Comma-separated steering strengths to sweep (e.g. "0,0.5,1.0,2.0")',
    )
    parser.add_argument(
        "--steering_apply_mode",
        choices=["last_prompt_and_current", "all_positions"],
        default="last_prompt_and_current",
        help="How to apply the steering vector (default: last_prompt_and_current)",
    )

    return parser


def run_mmlu_evaluation(args) -> Dict[str, Any]:
    """Run the MMLU-Redux evaluation for a parsed args namespace and return the
    results payload (a single run's payload, or a list of them for an alpha
    sweep). Contains everything the CLI used to do after parsing."""
    if args.batch_size is None:
        args.batch_size = 64 if args.backend in ("vllm", "openai") else 8
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.save_every_batches < 1:
        raise ValueError("--save_every_batches must be >= 1")
    if args.backend != "openai" and not args.model_path:
        raise ValueError("--model_path is required for the transformers/vllm backends.")
    if args.backend == "openai" and not args.base_url:
        raise ValueError("--backend openai requires --base_url (e.g. http://127.0.0.1:8100/v1).")

    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    if not alphas:
        alphas = [0.0]

    # vLLM + steering reloads the engine once per alpha (each evaluate() call builds
    # its own LLM). A second vLLM engine in the same process can fail to free the
    # first one's GPU memory and OOM, so for reliability run ONE process per alpha
    # (e.g. --alphas 0 then --alphas 34, with separate --output files).
    if args.backend == "vllm" and args.steering_direction_path and len(alphas) > 1:
        print(
            "NOTE: vLLM + steering with multiple alphas reloads the engine per alpha in "
            "one process (GPU-memory leak / OOM risk). For reliability, run one process "
            "per alpha with separate --output files."
        )

    results_all: List[Dict[str, Any]] = []
    for alpha in alphas:
        if args.output is not None and len(alphas) == 1:
            output_path = args.output
        else:
            model_ref = args.model_path or args.endpoint_model or args.base_model or "endpoint"
            model_short = model_ref.replace("/", "_").replace("\\", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            alpha_tag = f"_alpha{alpha}" if alpha != 0.0 else ""
            output_path = f"mmlu_redux_results_{model_short}{alpha_tag}_{ts}.json"

        if len(alphas) > 1:
            print(f"\n{'#' * 60}")
            print(f"# Steering alpha = {alpha}")
            print(f"{'#' * 60}")

        results_all.append(
            evaluate(
                model_path=args.model_path,
                base_model=args.base_model,
                backend=args.backend,
                subjects=args.subjects,
                num_shots=args.num_shots,
                system_prompt=args.system_prompt,
                max_new_tokens=args.max_new_tokens,
                disable_thinking=args.disable_thinking,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                seed=args.seed,
                dtype=args.dtype,
                output_path=output_path,
                gpu_memory_utilization=args.gpu_memory_utilization,
                save_responses=not args.no_save_responses,
                max_eval_examples_per_subject=args.max_eval_examples_per_subject,
                batch_size=args.batch_size,
                save_every_batches=args.save_every_batches,
                resume=args.resume,
                steering_direction_path=args.steering_direction_path,
                steering_layer=args.steering_layer,
                steering_alpha=alpha,
                steering_apply_mode=args.steering_apply_mode,
                base_url=args.base_url,
                api_key=args.api_key,
                endpoint_model=args.endpoint_model,
                endpoint_renderer=args.endpoint_renderer,
            )
        )

    return results_all[0] if len(results_all) == 1 else results_all


async def run_mmlu(
    *,
    client,
    base_model: str,
    output: Optional[str] = None,
    subjects: Optional[List[str]] = None,
    num_shots: int = 5,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    seed: int = 12345,
    save_responses: bool = True,
    max_eval_examples_per_subject: Optional[int] = None,
    chunk_size: int = 256,
) -> Dict[str, Any]:
    """Library-first MMLU-Redux eval through an injected ``ChatClient`` (no URL).

    The mirror of ``eval.run_evaluation`` for the capability-retention check: the
    client owns transport, caching, and concurrency; generation runs in a single
    event loop (chunked so pending tasks stay bounded). It reuses the shared
    item-building and scoring helpers, so accuracy matches the CLI path.
    Thinking-disabled is a property of the injected client's renderer (pass a
    disable-thinking client), matching the paper-facing MMLU protocol; ``top_k``
    ``-1`` is forwarded as "off" by the Tinker translation.
    """
    from generation import generate_openai

    subjects = subjects or ALL_SUBJECTS
    print(f"Loading MMLU-Redux data for {len(subjects)} subjects...")
    data = load_mmlu_redux(subjects)
    eval_items = build_eval_items(subjects, data, num_shots, max_eval_examples_per_subject)
    config = {
        "base_model": base_model,
        "backend": "openai",
        "num_shots": num_shots,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "disable_thinking": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "num_subjects": len(subjects),
        "subjects": subjects,
        "save_responses": save_responses,
        "dataset": "fxmarty/mmlu-redux-2.0-ok",
        "protocol": "Qwen3 tech report: 5-shot generative exact-match (in-process client)",
        "expected_total_questions": len(eval_items),
        "max_eval_examples_per_subject": max_eval_examples_per_subject,
    }

    t0 = time.time()
    per_question: List[Dict[str, Any]] = []
    for start in range(0, len(eval_items), chunk_size):
        chunk = eval_items[start:start + chunk_size]
        gens = await generate_openai(
            client,
            eval_prompts=[it["prompt"] for it in chunk],
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        for it, gen in zip(chunk, gens):
            per_question.append(build_per_question_record(it, gen["text"], save_responses))

    payload = build_results_payload(config, per_question, time.time() - t0)
    if output:
        write_results_atomic(output, payload)
    summ = payload["summary"]
    processed = summ["processed_questions"]
    return {
        "metrics": {
            "accuracy": summ["overall_accuracy"],
            "parse_failure_rate": (summ["total_parse_failures"] / processed) if processed else None,
        },
        "num_total": processed,
        "num_parse_failed": summ["total_parse_failures"],
        "output_path": output,
        "summary": summ,
    }


def run_mmlu_from_config(**overrides) -> Dict[str, Any]:
    """In-process entrypoint: build args with `build_parser` defaults, apply the
    given overrides, and run. The flow calls this instead of shelling out, so the
    endpoint and generation params pass as plain Python arguments end to end.

    Keyword names match the CLI flags without the leading dashes (e.g.
    ``base_url=..., endpoint_model=..., endpoint_renderer=..., subjects=...``)."""
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise TypeError(f"run_mmlu_from_config got unknown option {key!r}")
        setattr(args, key, value)
    return run_mmlu_evaluation(args)


def main():
    args = build_parser().parse_args()
    run_mmlu_evaluation(args)


if __name__ == "__main__":
    main()
