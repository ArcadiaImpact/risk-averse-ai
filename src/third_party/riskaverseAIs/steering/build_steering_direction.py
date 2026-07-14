"""Build a mean-difference steering direction for the risk-aversion experiments.

For each training situation we form two sequences that share the same prompt but
end with the risk-averse chain-of-thought (`chosen_full`) versus the
risk-neutral chain-of-thought (`rejected_full`). We capture the residual stream
at the output of a target decoder layer, pool over the response tokens, and
take the per-situation difference `chosen - rejected`.

The final steering vector is the mean of those per-situation differences
(CAA-mean / mean-difference activation steering). The saved direction is
L2-normalized to unit length, so `alpha` at evaluation time is the L2 magnitude
of the residual-stream perturbation.
"""

from __future__ import annotations

import argparse
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


def resolve_prompt(
    dataset_alias: str,
    base_model: str,
    explicit: str | None,
    *,
    force_default_system_prompt: bool,
) -> tuple[str, str]:
    """Resolve the canonical eval-time system prompt so direction build matches eval."""
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


def capture_layer_activation(model, layer_module, input_ids, prompt_len, position):
    """Return the pooled residual activation `(hidden_size,)` for one sequence."""
    holder = {}

    def hook(_module, _inp, out):
        holder["hidden"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()

    hidden = holder["hidden"][0]
    seq_len = hidden.shape[0]
    if position == "last":
        return hidden[seq_len - 1].float().cpu()
    start = min(prompt_len, seq_len - 1)
    return hidden[start:seq_len].mean(dim=0).float().cpu()


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
        "--layer",
        type=int,
        default=None,
        help="Decoder layer (default: middle layer).",
    )
    parser.add_argument("--position", choices=["mean_response", "last"], default="mean_response")
    parser.add_argument("--num_situations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max_len", type=int, default=8192, help="Truncate sequences to this many tokens.")
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Build with thinking OFF (matches a thinking-OFF evaluation).",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    layer = args.layer if args.layer is not None else n_layers // 2
    if not (0 <= layer < n_layers):
        raise SystemExit(f"--layer {layer} out of range (model has {n_layers} layers).")
    layer_module = layers[layer]
    print(f"Model has {n_layers} layers; extracting at layer {layer} ({args.position}).")

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

    diffs = []
    chosen_norms = []
    skipped = 0
    for i, row in df.iterrows():
        try:
            chosen_ids, prompt_len = build_ids(row["prompt_text"], row["chosen_full"])
            rejected_ids, _ = build_ids(row["prompt_text"], row["rejected_full"])
            if chosen_ids.shape[1] <= prompt_len or rejected_ids.shape[1] <= prompt_len:
                skipped += 1
                continue
            chosen = capture_layer_activation(model, layer_module, chosen_ids, prompt_len, args.position)
            rejected = capture_layer_activation(model, layer_module, rejected_ids, prompt_len, args.position)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            if skipped <= 3:
                print(f"  contrast {i} failed: {exc}")
            continue
        diffs.append(chosen - rejected)
        chosen_norms.append(chosen.norm().item())
        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(df)} (skipped {skipped})")

    if not diffs:
        raise SystemExit("No valid contrasts computed.")

    stacked = torch.stack(diffs)
    mean_diff = stacked.mean(dim=0)
    raw_norm = float(mean_diff.norm().item())
    unit = (mean_diff / raw_norm).float()
    per_diff_norms = stacked.norm(dim=-1)
    mean_resid_norm = float(sum(chosen_norms) / len(chosen_norms))

    steering_info = {
        "mode": "caa_mean",
        "method": "mean",
        "position": args.position,
        "extraction_layer": layer,
        "construction": "per_situation chosen_full vs rejected_full, same prompt",
    }
    payload = {
        "direction": unit,
        "vector": unit,
        "steering_info": steering_info,
        "method": "mean",
        "position": args.position,
        "layer": layer,
        "base_model": args.base_model,
        "hidden_size": int(unit.shape[0]),
        "num_situations": int(len(diffs)),
        "num_skipped": int(skipped),
        "seed": args.seed,
        "enable_thinking": not args.disable_thinking,
        "normalized": True,
        "raw_aggregate_norm": raw_norm,
        "mean_per_situation_diff_norm": float(per_diff_norms.mean().item()),
        "mean_residual_norm_at_layer": mean_resid_norm,
        "system_prompt": system_prompt,
        "training_csv": str(csv_path),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    print("\n" + "=" * 60)
    print("STEERING DIRECTION SAVED")
    print("=" * 60)
    print(f"output:                       {output_path}")
    print(f"method/position:              CAA-mean / {args.position}")
    print(f"layer:                        {layer} / {n_layers}")
    print(f"num situations (skipped):     {len(diffs)} ({skipped})")
    print(f"mean per-situation diff norm: {payload['mean_per_situation_diff_norm']:.3f}")
    print(f"mean residual norm @ layer:   {mean_resid_norm:.3f}")
    print("=" * 60)
    print("Alpha is the L2 magnitude of the perturbation (unit direction).")


if __name__ == "__main__":
    main()
