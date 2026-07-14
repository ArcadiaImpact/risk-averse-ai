# Reward-Model Fine-Tuning

This directory contains the reward-model fine-tuning (RMFT) pipeline used in
the paper. It trains an `AutoModel` backbone with a LoRA adapter plus a
separately trained scalar reward head, then evaluates the resulting checkpoint
on the held-out risk-aversion pairwise tasks and on RewardBench 2.

The pipeline reads its datasets from the shared `../evaluation/data/` directory
and uses `../evaluation/evaluate_reward_model.py` for canonical held-out scoring.

## Install

```bash
pip install -r requirements.txt
```

## Locked Qwen3-8B Run

The paper-facing Qwen3-8B RMFT configuration uses:

- learning rate: `5e-4`
- epochs: `5`
- weight decay: `0.05`
- batch / grad-accum: `2 / 32`
- shared 7-module LoRA geometry
- no system prompt during training or reward-model evaluation

Run the locked configuration across the held-out seeds:

```bash
python rft_pipeline.py \
  --base_model Qwen/Qwen3-8B \
  --middle_lr 5e-4 \
  --single_lr \
  --epochs 5
```

This runs:

1. validation at the locked learning rate
2. held-out evaluation for seeds `1,2,3`
3. RewardBench 2 for the held-out checkpoints

Outputs are written under `outputs/`.

## Useful Variants

Single-seed smoke:

```bash
python rft_pipeline.py \
  --base_model Qwen/Qwen3-8B \
  --middle_lr 5e-4 \
  --single_lr \
  --epochs 5 \
  --seeds_validation 1 \
  --seeds_heldout 1
```

Evaluate an untrained random-head baseline:

```bash
python rft_pipeline.py \
  --base_model Qwen/Qwen3-8B \
  --middle_lr 5e-4 \
  --single_lr \
  --epochs 5 \
  --no_train \
  --seeds_validation 1 \
  --seeds_heldout 1
```

## Held-Out Datasets

The canonical held-out reward-model aliases are:

- `reward_model_validation`
- `reward_model_high_stakes_test`
- `reward_model_astronomical_stakes_deployment`
- `reward_model_steals_test`

The canonical reward-model evaluator is:

```bash
python ../evaluation/evaluate_reward_model.py --list_datasets
```

## Regenerating Held-Out CoTs (Optional)

The held-out reward-model CSVs ship pre-generated in `../evaluation/data/`, so
normal reproduction never needs this step. `generate_val_cots.py` is the script
that produced their chosen/rejected CoT pairs, using Claude Sonnet. Running it
requires `pip install anthropic` (not in `requirements.txt`) and an
`ANTHROPIC_API_KEY` environment variable, and regenerated CoTs will not exactly
match the shipped ones.
