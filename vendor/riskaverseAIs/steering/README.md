# Steering

This directory contains the paper's steering workflow. The paper-facing method
is mean-difference activation steering (CAA-mean), and steering is evaluated in
thinking-off mode.

## Install

There is no separate requirements file for this directory. The steering scripts
import from the shared evaluation package, so install the evaluation
requirements first:

```bash
pip install -r ../evaluation/requirements.txt
```

## Build a Steering Direction

The direction builder uses the 600-row lin-only low-stakes CoT file. The locked
Qwen3-8B configuration captures the direction at layer 18:

```bash
python build_steering_direction.py \
  --base_model Qwen/Qwen3-8B \
  --training_csv ../evaluation/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv \
  --dataset_alias medium_stakes_validation \
  --position mean_response \
  --layer 18 \
  --num_situations 200 \
  --seed 12345 \
  --output steering_qwen3_8b.pt
```

If `--layer` is omitted, the builder defaults to the model's middle layer
(layer 18 for the 36-layer Qwen3-8B, so the default matches the locked
configuration for that model).

To build directions for many layers in one pass (useful for layer sweeps), use
`build_steering_directions_multi.py`, which loads the model once and writes one
direction file per requested layer.

## Locked Qwen3-8B Evaluation

The locked Qwen3-8B paper configuration is:

- direction construction: `CAA-mean`
- eval layer: `18`
- steering strength: `34`
- thinking: off

Evaluate it with the shared evaluator:

```bash
python ../evaluation/evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --backend vllm \
  --disable_thinking \
  --steering_direction_path steering_qwen3_8b.pt \
  --eval_layer 18 \
  --alphas 34 \
  --output qwen3_8b_steering_medium_val.json
```

For the held-out runs, swap the dataset alias to:

- `high_stakes_test`
- `astronomical_stakes_deployment`
- `steals_test`

## Capability Retention

The same steering artifact can be passed to `../evaluation/evaluate_mmlu_redux.py`
with `--steering_direction_path`, `--steering_layer`, and `--alphas`.
