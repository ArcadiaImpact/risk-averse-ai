# SFT Training

This directory contains the supervised fine-tuning code used for the paper's
policy-training runs. The same trainer is also used for the supervised tie
training variant in [`../tie-training`](../tie-training).

## Data

- Standard CoT training data:
  - `data/CoT-training/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv`
  - `data/CoT-training/2026_04_13_tie_training_modified_CoTs.csv`
- No-think-tag copies for model families trained without Qwen-style `<think>` tags:
  - `data/NO_THINK_TAGS/2026_03_22_low_stakes_training_set_1000_situations_CoTs_no_think_tags.csv`
  - `data/NO_THINK_TAGS/2026_04_13_tie_training_modified_CoTs_no_think_tags.csv`

## Install

```bash
pip install -r requirements.txt
```

## Locked Qwen3-8B SFT Run

This reproduces the paper-facing Qwen3-8B SFT configuration: 1000 low-stakes
CoT training examples, 4 epochs, learning rate `5e-4`, batch / grad-accum
`4 / 4`, shared 7-module LoRA, and the shared evaluator.

```bash
python train_and_evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --learning_rate 5e-4 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --cot_unmodified_train_examples 1000 \
  --cot_modified_train_examples 0 \
  --run_name qwen3_8b_sft \
  --eval_script ../evaluation/evaluate.py \
  --eval_datasets medium_stakes_validation high_stakes_test astronomical_stakes_deployment steals_test
```

## Llama / Gemma SFT Run

For Llama and Gemma, use the no-think-tag copies and an empty system prompt.

```bash
python train_and_evaluate.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --system_prompt "" \
  --unmodified_cot_data data/NO_THINK_TAGS/2026_03_22_low_stakes_training_set_1000_situations_CoTs_no_think_tags.csv \
  --modified_cot_data data/NO_THINK_TAGS/2026_04_13_tie_training_modified_CoTs_no_think_tags.csv \
  --cot_unmodified_train_examples 1000 \
  --cot_modified_train_examples 0 \
  --learning_rate 1e-4 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --run_name llama31_8b_sft \
  --eval_script ../evaluation/evaluate.py \
  --eval_datasets medium_stakes_validation high_stakes_test astronomical_stakes_deployment steals_test
```

For Gemma, swap the model ID and keep `--system_prompt ""`.
