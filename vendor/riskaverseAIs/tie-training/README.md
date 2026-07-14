# Tie Training

The paper's tie-training method is a supervised variant built on the same
trainer as SFT. The only change is that some of the 1000 low-stakes training
examples are replaced with tie examples from the modified CoT dataset while the
total corpus size stays fixed.

The implementation lives in [`../sft-training/train_and_evaluate.py`](../sft-training/train_and_evaluate.py).

## Locked Qwen3-8B Tie-Training Run

The locked Qwen3-8B tie rate is `30%`, implemented as `700` unmodified examples
plus `300` modified examples. The rest of the training configuration matches
the locked SFT setup.

```bash
cd ../sft-training

python train_and_evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --learning_rate 5e-4 \
  --num_train_epochs 4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --cot_unmodified_train_examples 700 \
  --cot_modified_train_examples 300 \
  --run_name qwen3_8b_tie30 \
  --eval_script ../evaluation/evaluate.py \
  --eval_datasets medium_stakes_validation high_stakes_test astronomical_stakes_deployment steals_test
```

## Llama / Gemma Tie Training

Use the same command pattern, but point `--unmodified_cot_data` and
`--modified_cot_data` at the no-think-tag copies in
`../sft-training/data/NO_THINK_TAGS/`, and set `--system_prompt ""`.
