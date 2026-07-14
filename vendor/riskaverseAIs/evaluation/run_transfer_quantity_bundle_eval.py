import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import evaluate as ev
from dataset_schema_utils import ensure_option_level_dataframe


TRANSFER_DATASETS = [
    "gpu_hours_transfer_benchmark",
    "lives_saved_transfer_benchmark",
    "money_for_user_transfer_benchmark",
]

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--datasets", nargs="+", default=list(TRANSFER_DATASETS))
    parser.add_argument("--num_situations", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--reasoning_max_tokens", type=int, default=800)
    parser.add_argument("--max_time_per_generation", type=float, default=300.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--backup_every", type=int, default=0)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_model_len", type=int, default=None)
    parser.add_argument("--vllm_dtype", default="auto")
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64)
    parser.add_argument("--vllm_enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--no_vllm_enable_prefix_caching", dest="vllm_enable_prefix_caching", action="store_false")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--prompt_suffix", default="")
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


class _TolerantNamespace(SimpleNamespace):
    """SimpleNamespace that returns None for attributes that were never set.

    evaluate.py's arg surface grows over time. This bundle wrapper builds the
    args object by hand, so any newer optional flag evaluate.py references but
    we do not set would otherwise raise AttributeError. Defaulting unknown
    attributes to None matches argparse's default=None / store_true-off
    semantics for those optional features.
    """

    def __getattr__(self, name):  # only called when normal attribute lookup fails
        return None


def build_args(
    base_args,
    *,
    dataset: str,
    csv_path: str,
    resolved_variant: str,
    dataset_base_alias: str,
    system_prompt: str,
    system_prompt_source: str,
    output_path: str,
):
    return _TolerantNamespace(
        backend=base_args.backend,
        base_model=base_args.base_model,
        model_path=base_args.model_path,
        dataset=dataset,
        dataset_base_alias=dataset_base_alias,
        resolved_dataset_variant=resolved_variant,
        dataset_variant="default",
        custom_csv=None,
        csv_path=csv_path,
        lin_only=False,
        num_situations=base_args.num_situations,
        temperature=base_args.temperature,
        top_p=base_args.top_p,
        top_k=base_args.top_k,
        seed=base_args.seed,
        max_new_tokens=base_args.max_new_tokens,
        reasoning_max_tokens=base_args.reasoning_max_tokens,
        batch_size=base_args.batch_size,
        max_time_per_generation=base_args.max_time_per_generation,
        system_prompt=system_prompt,
        prompt_suffix=base_args.prompt_suffix,
        disable_thinking=base_args.disable_thinking,
        save_every=base_args.save_every,
        backup_every=base_args.backup_every,
        no_save_responses=False,
        steering_apply_mode="last_prompt_and_current",
        start_position=1,
        end_position=base_args.num_situations,
        stop_after=base_args.num_situations,
        resume=base_args.resume,
        output=output_path,
        alphas="0.0",
        system_prompt_source=system_prompt_source,
        vllm_tensor_parallel_size=base_args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=base_args.vllm_gpu_memory_utilization,
        vllm_max_model_len=base_args.vllm_max_model_len,
        vllm_dtype=base_args.vllm_dtype,
        vllm_enable_prefix_caching=base_args.vllm_enable_prefix_caching,
        vllm_max_lora_rank=base_args.vllm_max_lora_rank,
    )


def load_situations(dataset: str, num_situations: int):
    csv_path, resolved_variant, dataset_base_alias = ev.resolve_builtin_dataset_path(dataset, "default")
    df = pd.read_csv(csv_path)
    df = ensure_option_level_dataframe(df)
    ev.validate_dataset_columns(df, csv_path)
    all_situations = ev.build_situations(df, num_situations)
    situations = all_situations[:num_situations]
    return situations, csv_path, resolved_variant, dataset_base_alias


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.backend == "vllm":
        torch.default_generator.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = SimpleNamespace(
        base_model=args.base_model,
        model_path=args.model_path,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_dtype=args.vllm_dtype,
        vllm_enable_prefix_caching=args.vllm_enable_prefix_caching,
        vllm_max_lora_rank=args.vllm_max_lora_rank,
    )

    overall_start = time.time()
    model, lora_request = ev.load_vllm_engine(model_args)
    per_dataset = []

    try:
        for dataset in args.datasets:
            dataset_start = time.time()
            situations, csv_path, resolved_variant, dataset_base_alias = load_situations(dataset, args.num_situations)
            resolved_system_prompt, system_prompt_source = ev.resolve_system_prompt(
                dataset_base_alias=dataset_base_alias,
                base_model=args.base_model,
                model_path=args.model_path,
                explicit_system_prompt=args.system_prompt,
            )
            output_path = output_dir / (
                f"2026_04_11_{dataset}_{args.base_model.replace('/', '_').replace('.', '_').lower()}_bundle_n{args.num_situations}.json"
            )
            run_args = build_args(
                args,
                dataset=dataset,
                csv_path=csv_path,
                resolved_variant=resolved_variant,
                dataset_base_alias=dataset_base_alias,
                system_prompt=resolved_system_prompt,
                system_prompt_source=system_prompt_source,
                output_path=str(output_path),
            )
            summary = ev.run_single_alpha_eval(
                backend="vllm",
                model=model,
                tokenizer=None,
                situations=situations,
                args=run_args,
                output_path=str(output_path),
                steering_alpha=0.0,
                steering_info=None,
                steering_block=None,
                steering_direction=None,
                lora_request=lora_request,
            )
            dataset_elapsed = time.time() - dataset_start
            per_dataset.append(
                {
                    "dataset": dataset,
                    "csv_path": csv_path,
                    "num_situations": len(situations),
                    "system_prompt_source": system_prompt_source,
                    "output_path": str(output_path),
                    "wall_seconds": dataset_elapsed,
                    "summary": summary,
                }
            )
    finally:
        del model
        gc.collect()

    manifest = {
        "backend": args.backend,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "datasets": args.datasets,
        "num_situations": args.num_situations,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_max_tokens": args.reasoning_max_tokens,
        "enable_thinking": not args.disable_thinking,
        "system_prompt": args.system_prompt,
        "system_prompt_source": (
            "cli_system_prompt" if args.system_prompt is not None else "resolved_per_dataset"
        ),
        "total_wall_seconds": time.time() - overall_start,
        "per_dataset": per_dataset,
    }
    manifest_path = output_dir / (
        f"2026_04_11_transfer_bundle_manifest_{args.base_model.replace('/', '_').replace('.', '_').lower()}_n{args.num_situations}.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
