# Evaluation

> **Provenance.** This tree is lifted from the riskaverseAIs benchmark's
> `evaluation/` subtree @ `79f2da1` (see `src/third_party/README.md`) and is
> maintained first-party: local modifications are allowed and tracked by git,
> so it may diverge from upstream. Licensing: code MIT, datasets CC-BY-4.0 —
> the license texts live once, in
> `src/third_party/riskaverseAIs/` (`LICENSE`,
> `sft-training/LICENSE-CC-BY-4.0.txt`, `sft-training/DATA_LICENSE.md`),
> and cover this subtree.

This is a **subset** of the upstream evaluation suite — the parts the
risk-averse-ai flow actually exercises: the main generative policy benchmark,
the transfer-quantity benchmarks, and the MMLU-Redux capability-retention
check. The reward-model evaluation suite, the multi-dataset bundle runners, and
the dataset re-generation scripts are not vendored here; find them in the
upstream release.

## Contents

- `evaluate.py` — main generative policy evaluation (and its `answer_parser`,
  `dataset_schema_utils`, `risk_averse_prompts` helpers).
- `evaluate_mmlu_redux.py` — MMLU-Redux capability-retention evaluation.
- `data/` — the paper-facing CSVs these evaluations read (see below).
- `tests/` — unit tests for the surviving modules.

The flow drives `evaluate.py` and `evaluate_mmlu_redux.py` in-process against a
local OpenAI-compatible shim; a vLLM backend is also retained as the parity
anchor.

## Install

```bash
pip install -r requirements.txt
```

## Main Generative Evaluation

List the built-in datasets:

```bash
python evaluate.py --list_datasets
```

Evaluate a base model or adapter on the main policy benchmark:

```bash
python evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --backend vllm \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 12345 \
  --batch_size 4 \
  --max_new_tokens 4096 \
  --reasoning_max_tokens 800 \
  --output qwen3_8b_medium_val.json
```

To evaluate an adapter, add `--model_path /path/to/adapter`.

Built-in policy dataset aliases:

- `medium_stakes_validation`
- `high_stakes_test`
- `astronomical_stakes_deployment`
- `steals_test`
- `low_stakes_training`
- `low_stakes_validation`
- `low_stakes_training_lin_only`
- `low_stakes_validation_lin_only`
- `gpu_hours_transfer_benchmark`
- `lives_saved_transfer_benchmark`
- `money_for_user_transfer_benchmark`

Qwen models use the shared default system prompt with thinking enabled. Llama
and Gemma runs default to no system prompt.

## Capability Retention

`evaluate_mmlu_redux.py` runs the paper's MMLU-Redux evaluation protocol. The
paper-facing settings are 5-shot, deterministic decoding, and thinking disabled.

```bash
python evaluate_mmlu_redux.py \
  --model_path /path/to/model_or_adapter \
  --base_model Qwen/Qwen3-8B \
  --backend vllm \
  --disable_thinking \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k -1 \
  --min_p 0.0 \
  --output mmlu_redux.json
```

## Data

The paper-facing CSVs under `data/` used by the evaluations and the
benchmark-recipe training arms:

- the main validation / test / deployment / steals evaluation sets
- the transfer-to-other-quantities benchmarks (`transfer_to_other_quantities/`)
- low-stakes CoT training data and the 600-row lin-only variant
- tie-training CSVs
- no-think-tag variants for Llama / Gemma runs (`NO_THINK_TAGS/`)

The reward-model evaluation CSVs are not vendored here; find them in the
upstream release.
