#!/usr/bin/env python3
"""Run the four paper-facing OOD risk-aversion evals via the standard evaluate.py CLI."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import evaluate as ev


MAIN_DATASETS = [
    "medium_stakes_validation",
    "high_stakes_test",
    "astronomical_stakes_deployment",
    "steals_test",
]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--datasets", nargs="+", default=list(MAIN_DATASETS), choices=sorted(ev.DATASET_ALIASES.keys()))
    parser.add_argument(
        "--num_situations",
        type=int,
        default=None,
        help=(
            "Optional global override for the number of situations per selected dataset. "
            "If omitted, each dataset uses the repo's current recommended default."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=ev.DEFAULT_EVAL_TEMPERATURE)
    parser.add_argument(
        "--allow_nondefault_temperature",
        action="store_true",
        help=(
            "Required when using a temperature different from the canonical paper default "
            f"of {ev.DEFAULT_EVAL_TEMPERATURE}."
        ),
    )
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--reasoning_max_tokens", type=int, default=800)
    parser.add_argument("--max_time_per_generation", type=float, default=300.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_every", type=int, default=4)
    parser.add_argument("--backup_every", type=int, default=20)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_dtype", default="auto")
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64)
    parser.add_argument(
        "--vllm_enable_prefix_caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument(
        "--no_system_prompt",
        action="store_true",
        help="Pass an explicitly empty system prompt through to evaluate.py.",
    )
    parser.add_argument("--prompt_suffix", default="")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "_").replace("-", "_").lower()


def resolve_dataset_num_situations(dataset: str, override: int | None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("--num_situations must be >= 1 when provided.")
        return override

    default_num = ev.DEFAULT_NUM_SITUATIONS_BY_DATASET.get(dataset)
    if default_num is None:
        raise ValueError(f"No recommended default number of situations is configured for dataset {dataset!r}.")
    return int(default_num)


def build_eval_command(args, *, dataset: str, num_situations: int, output_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "evaluate.py",
        "--backend",
        args.backend,
        "--base_model",
        args.base_model,
        "--dataset",
        dataset,
        "--num_situations",
        str(num_situations),
        "--batch_size",
        str(args.batch_size),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--reasoning_max_tokens",
        str(args.reasoning_max_tokens),
        "--max_time_per_generation",
        str(args.max_time_per_generation),
        "--save_every",
        str(args.save_every),
        "--backup_every",
        str(args.backup_every),
        "--vllm_tensor_parallel_size",
        str(args.vllm_tensor_parallel_size),
        "--vllm_gpu_memory_utilization",
        str(args.vllm_gpu_memory_utilization),
        "--vllm_dtype",
        str(args.vllm_dtype),
        "--vllm_max_lora_rank",
        str(args.vllm_max_lora_rank),
        "--output",
        str(output_path),
    ]
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if abs(args.temperature - ev.DEFAULT_EVAL_TEMPERATURE) > 1e-12:
        cmd.append("--allow_nondefault_temperature")
    if args.resume:
        cmd.append("--resume")
    if args.disable_thinking:
        cmd.append("--disable_thinking")
    if args.vllm_max_model_len is not None:
        cmd.extend(["--vllm_max_model_len", str(args.vllm_max_model_len)])
    if args.vllm_enable_prefix_caching:
        cmd.append("--vllm_enable_prefix_caching")
    else:
        cmd.append("--no-vllm_enable_prefix_caching")
    if args.system_prompt is not None:
        cmd.extend(["--system_prompt", args.system_prompt])
    if args.prompt_suffix:
        cmd.extend(["--prompt_suffix", args.prompt_suffix])
    return cmd


def summarize_output_payload(output_path: Path) -> dict:
    payload = json.loads(output_path.read_text())
    return {
        "output_path": str(output_path),
        "alpha": 0.0,
        "metrics": payload["metrics"],
        "num_valid": payload["num_valid"],
        "num_behaviorally_classified": payload.get("num_behaviorally_classified"),
        "num_total": payload["num_total"],
        "num_parse_failed": payload["num_parse_failed"],
        "num_resumed": payload.get("evaluation_config", {}).get("num_situations_completed", 0)
        - len(payload.get("results", [])),
        "num_new": len(payload.get("results", [])),
    }


def main():
    parser = build_parser()
    args = parser.parse_args()

    if abs(args.temperature - ev.DEFAULT_EVAL_TEMPERATURE) > 1e-12 and not args.allow_nondefault_temperature:
        raise ValueError(
            "Non-default --temperature requested "
            f"({args.temperature}). The canonical paper eval default is {ev.DEFAULT_EVAL_TEMPERATURE}. "
            "If you really intend to run off-default, re-run with --allow_nondefault_temperature."
        )
    if args.no_system_prompt and args.system_prompt is not None:
        raise ValueError("Use either --system_prompt or --no_system_prompt, not both.")
    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")
    if args.backup_every < 0:
        raise ValueError("--backup_every must be >= 0")
    if args.no_system_prompt:
        args.system_prompt = ""

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_tag = slugify_model_name(args.model_path or args.base_model)
    overall_start = time.time()
    per_dataset = []

    for dataset in args.datasets:
        num_situations = resolve_dataset_num_situations(dataset, args.num_situations)
        output_path = output_dir / f"{dataset}_{model_tag}.json"
        cmd = build_eval_command(args, dataset=dataset, num_situations=num_situations, output_path=output_path)

        dataset_start = time.time()
        print("\n" + "=" * 80)
        print(f"Starting dataset: {dataset} ({num_situations} situations)")
        print("=" * 80)
        print("RUNNING:", " ".join(cmd))

        completed = subprocess.run(
            cmd,
            cwd=REPO_DIR,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{dataset} failed with exit code {completed.returncode}.")
        if not output_path.exists():
            raise FileNotFoundError(f"{dataset} finished without producing {output_path}.")

        dataset_elapsed = time.time() - dataset_start
        payload = json.loads(output_path.read_text())
        eval_config = payload.get("evaluation_config", {})
        per_dataset.append(
            {
                "dataset": dataset,
                "csv_path": eval_config.get("csv_path"),
                "resolved_dataset_variant": eval_config.get("dataset_variant"),
                "dataset_base_alias": eval_config.get("dataset_base_alias"),
                "system_prompt_source": eval_config.get("system_prompt_source"),
                "num_situations": eval_config.get("num_situations"),
                "output_path": str(output_path),
                "wall_seconds": dataset_elapsed,
                "summary": summarize_output_payload(output_path),
            }
        )

    manifest = {
        "backend": args.backend,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "datasets": args.datasets,
        "requested_num_situations_override": args.num_situations,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "max_time_per_generation": args.max_time_per_generation,
        "save_every": args.save_every,
        "backup_every": args.backup_every,
        "enable_thinking": not args.disable_thinking,
        "system_prompt": args.system_prompt,
        "system_prompt_source": (
            "cli_system_prompt" if args.system_prompt is not None else "deferred_to_evaluate.py"
        ),
        "prompt_suffix": args.prompt_suffix,
        "save_responses": True,
        "resume": args.resume,
        "vllm": {
            "tensor_parallel_size": args.vllm_tensor_parallel_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "max_model_len": args.vllm_max_model_len,
            "dtype": args.vllm_dtype,
            "enable_prefix_caching": args.vllm_enable_prefix_caching,
            "max_lora_rank": args.vllm_max_lora_rank if args.model_path else None,
        },
        "total_wall_seconds": time.time() - overall_start,
        "per_dataset": per_dataset,
    }
    manifest_path = output_dir / f"ood_risk_eval_bundle_manifest_{model_tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
