"""Build mean-difference steering directions at many layers in one pass.

This is the multi-layer companion to `build_steering_direction.py`. It loads
the model once, captures residual activations at every requested layer during
the same forward passes, and writes one CAA-mean direction file per layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch


STEERING_DIR = Path(__file__).resolve().parent
REPO_DIR = STEERING_DIR.parent
EVALUATION_DIR = REPO_DIR / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from evaluate import apply_chat_template_safe, get_decoder_layers
from risk_averse_prompts import default_system_prompt_for_dataset, resolve_system_prompt


DEFAULT_TRAIN_CSV = (
    "evaluation/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv"
)
OUTPUT_LABEL = "caa_mean"


def resolve_prompt(
    dataset_alias: str,
    base_model: str,
    explicit: str | None,
    *,
    force_default_system_prompt: bool,
) -> tuple[str, str]:
    if explicit is not None:
        return explicit, "cli"
    if force_default_system_prompt:
        return default_system_prompt_for_dataset(dataset_alias), "forced_default_system_prompt"
    try:
        resolved, source = resolve_system_prompt(
            dataset_base_alias=dataset_alias,
            base_model=base_model,
            model_path=None,
            explicit_system_prompt=None,
        )
        return resolved, source
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Could not resolve the canonical system prompt for alias {dataset_alias!r}: {exc}. "
            "Pass --system_prompt_file with the eval-time system prompt."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--training_csv", default=DEFAULT_TRAIN_CSV)
    parser.add_argument(
        "--dataset_alias",
        default="medium_stakes_validation",
        help="Alias used only to resolve the canonical system prompt.",
    )
    parser.add_argument("--system_prompt_file", default=None)
    parser.add_argument("--force_default_system_prompt", action="store_true")
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated decoder layer indices, e.g. '8,12,16,20,24,28'.",
    )
    parser.add_argument("--position", choices=["mean_response", "last"], default="mean_response")
    parser.add_argument("--num_situations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--output_template",
        default="dir_caa_mean_L{layer}.pt",
        help="Filename template; {layer} is the decoder-layer index.",
    )
    parser.add_argument(
        "--summary_json",
        default=None,
        help="Optional path to write a {layer: {diff_norm, resid_norm}} summary map.",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if not target_layers:
        raise SystemExit("Provide at least one layer in --layers.")

    if args.force_default_system_prompt and args.system_prompt_file:
        raise SystemExit("Use either --system_prompt_file or --force_default_system_prompt, not both.")

    explicit_sp = None
    if args.system_prompt_file:
        explicit_sp = Path(args.system_prompt_file).read_text().strip()
    system_prompt, sp_source = resolve_prompt(
        args.dataset_alias,
        args.base_model,
        explicit_sp,
        force_default_system_prompt=args.force_default_system_prompt,
    )
    print(f"System prompt source: {sp_source} ({len(system_prompt)} chars)")

    csv_path = Path(args.training_csv)
    if not csv_path.is_absolute() and not csv_path.exists():
        csv_path = REPO_DIR / csv_path
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "rejected_type" in df.columns:
        df = df[df["rejected_type"] == "lin"]
    df = df.dropna(subset=["prompt_text", "chosen_full", "rejected_full"]).reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True).head(args.num_situations)
    print(f"Using {len(df)} situations from {csv_path}")

    print(f"Loading {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    layers = get_decoder_layers(model)
    n_layers = len(layers)
    for layer in target_layers:
        if not (0 <= layer < n_layers):
            raise SystemExit(f"--layers includes {layer}, out of range (model has {n_layers} layers).")
    print(f"Model has {n_layers} layers; extracting at {target_layers} ({args.position}).")

    holder: dict[int, torch.Tensor] = {}

    def make_hook(idx):
        def hook(_module, _inp, out):
            holder[idx] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    handles = [layers[idx].register_forward_hook(make_hook(idx)) for idx in target_layers]

    def build_ids(prompt_text, response_text):
        prefix = apply_chat_template_safe(
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            disable_thinking=args.disable_thinking,
        )
        prompt_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        full_ids = tokenizer(prefix + str(response_text), add_special_tokens=False).input_ids
        full_ids = full_ids[: args.max_len]
        return torch.tensor([full_ids], device=device), len(prompt_ids)

    def capture_all(input_ids, prompt_len):
        holder.clear()
        with torch.no_grad():
            model(input_ids=input_ids)
        out = {}
        for idx in target_layers:
            hidden = holder[idx][0]
            seq_len = hidden.shape[0]
            if args.position == "last":
                out[idx] = hidden[seq_len - 1].float().cpu()
            else:
                start = min(prompt_len, seq_len - 1)
                out[idx] = hidden[start:seq_len].mean(dim=0).float().cpu()
        return out

    diffs = {idx: [] for idx in target_layers}
    chosen_norms = {idx: [] for idx in target_layers}
    skipped = 0
    try:
        for i, row in df.iterrows():
            try:
                chosen_ids, prompt_len = build_ids(row["prompt_text"], row["chosen_full"])
                rejected_ids, _ = build_ids(row["prompt_text"], row["rejected_full"])
                if chosen_ids.shape[1] <= prompt_len or rejected_ids.shape[1] <= prompt_len:
                    skipped += 1
                    continue
                chosen = capture_all(chosen_ids, prompt_len)
                rejected = capture_all(rejected_ids, prompt_len)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                if skipped <= 3:
                    print(f"  contrast {i} failed: {exc}")
                continue
            for idx in target_layers:
                diffs[idx].append(chosen[idx] - rejected[idx])
                chosen_norms[idx].append(chosen[idx].norm().item())
            if (i + 1) % 25 == 0:
                print(f"  processed {i + 1}/{len(df)} (skipped {skipped})")
    finally:
        for handle in handles:
            handle.remove()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("PER-LAYER SUMMARY")
    print("=" * 60)
    print(f"{'layer':>5} | {'diff_norm':>9} | {'resid_norm':>10}")
    print("-" * 60)

    summary: dict[str, dict] = {}
    for idx in target_layers:
        if not diffs[idx]:
            print(f"{idx:>5} | (no valid contrasts; skipped)")
            continue

        stacked = torch.stack(diffs[idx])
        mean_diff = stacked.mean(dim=0)
        raw_norm = float(mean_diff.norm().item())
        unit = (mean_diff / raw_norm).float()
        per_diff_norm = float(stacked.norm(dim=-1).mean().item())
        mean_resid_norm = float(sum(chosen_norms[idx]) / len(chosen_norms[idx]))

        payload = {
            "direction": unit,
            "vector": unit,
            "steering_info": {
                "mode": OUTPUT_LABEL,
                "method": "mean",
                "position": args.position,
                "extraction_layer": idx,
                "construction": "per_situation chosen_full vs rejected_full, same prompt",
            },
            "method": "mean",
            "position": args.position,
            "layer": idx,
            "base_model": args.base_model,
            "hidden_size": int(unit.shape[0]),
            "num_situations": int(len(diffs[idx])),
            "num_skipped": int(skipped),
            "seed": args.seed,
            "enable_thinking": not args.disable_thinking,
            "normalized": True,
            "raw_aggregate_norm": raw_norm,
            "mean_per_situation_diff_norm": per_diff_norm,
            "mean_residual_norm_at_layer": mean_resid_norm,
            "system_prompt": system_prompt,
            "training_csv": str(csv_path),
        }

        out_name = args.output_template.format(layer=idx)
        out_path = output_dir / out_name
        torch.save(payload, out_path)
        summary[str(idx)] = {
            "path": str(out_path),
            "diff_norm": per_diff_norm,
            "resid_norm": mean_resid_norm,
        }
        print(f"{idx:>5} | {per_diff_norm:>9.3f} | {mean_resid_norm:>10.3f}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
