# DPO Training

This directory contains the DPO trainer used for the paper's preference-tuning
experiments.

## Data

The paper-facing DPO runs use only the `600` lin-only low-stakes pairs:

- Qwen:
  - `../evaluation/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv`
- Llama / Gemma no-think-tag copy:
  - `../evaluation/data/NO_THINK_TAGS/2026_03_22_low_stakes_training_set_600_situations_CoTs_no_think_tags_lin_only.csv`

## Install

```bash
pip install -r requirements.txt
```

## Build the JSONL

Qwen:

```bash
python prepare_dpo_dataset.py \
  --input_csv ../evaluation/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv \
  --output_jsonl data/qwen3_8b_strict_dpo.jsonl
```

Llama / Gemma:

```bash
python prepare_dpo_dataset.py \
  --input_csv ../evaluation/data/NO_THINK_TAGS/2026_03_22_low_stakes_training_set_600_situations_CoTs_no_think_tags_lin_only.csv \
  --output_jsonl data/llama_or_gemma_strict_dpo.jsonl
```

## Locked Qwen3-8B Run

The locked Qwen3-8B DPO configuration uses 4-bit QLoRA, the shared 7-module
LoRA geometry, learning rate `1e-4`, `beta=0.10`, `3` epochs, batch / grad-accum
`2 / 8`, the shared default system prompt, and Qwen thinking enabled.

```bash
python train_dpo_lora.py \
  --model_name Qwen/Qwen3-8B \
  --train_path data/qwen3_8b_strict_dpo.jsonl \
  --learning_rate 1e-4 \
  --beta 0.10 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --run_name qwen3_8b_dpo \
  --output_root runs
```

## Llama / Gemma Run

For Llama and Gemma, keep the same trainer settings but use an empty system
prompt and the no-think-tag JSONL:

```bash
python train_dpo_lora.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --system_prompt "" \
  --train_path data/llama_or_gemma_strict_dpo.jsonl \
  --learning_rate 1e-4 \
  --beta 0.10 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --run_name llama31_8b_dpo \
  --output_root runs
```

## Evaluate a Trained Adapter

```bash
python ../evaluation/evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --model_path runs/qwen3_8b_dpo/final_adapter \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --backend vllm \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --batch_size 4 \
  --max_new_tokens 4096 \
  --reasoning_max_tokens 800 \
  --output qwen3_8b_dpo_medium_val.json
```
