# Evaluation

Shared evaluation package for the paper's policy models, reward models,
transfer benchmarks, and capability-retention checks.

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

## Reward-Model Evaluation

The reward-model evaluator is separate because it scores chosen vs rejected CoTs
rather than generating answers.

List reward-model datasets:

```bash
python evaluate_reward_model.py --list_datasets
```

Evaluate a reward-model checkpoint:

```bash
python evaluate_reward_model.py \
  --base_model Qwen/Qwen3-8B \
  --model_path /path/to/checkpoint \
  --dataset reward_model_validation \
  --output reward_model_validation.json
```

Built-in reward-model dataset aliases:

- `reward_model_validation`
- `reward_model_high_stakes_test`
- `reward_model_astronomical_stakes_deployment`
- `reward_model_steals_test`

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

## Steering

The paper's activation-steering workflow lives in [`../steering/README.md`](../steering/README.md).
Use `evaluate.py` to run a precomputed steering direction with one or more
`--alphas`.

## Data

The canonical paper-facing CSVs are under `data/`:

- low-stakes CoT training data
- the 600-row lin-only DPO / steering source file
- the main validation / test / deployment / steals evaluation sets
- reward-model evaluation CSVs
- tie-training CSVs
- no-think-tag variants for Llama / Gemma runs
- transfer-to-other-quantities benchmarks

The paper tables use these pre-generated CSVs directly. Regeneration scripts for
the main and transfer datasets are included in `../dataset-generation/`.
