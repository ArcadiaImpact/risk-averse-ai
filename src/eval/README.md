# Evaluation

> **Provenance.** Lifted from the riskaverseAIs benchmark's `evaluation/`
> subtree @ `79f2da1` (see `src/third_party/README.md`) and maintained
> first-party — local modifications are tracked by git and may diverge from
> upstream; modules beyond what this repo exercises live in the upstream
> release. Licensing: code MIT, datasets CC-BY-4.0; the license texts live
> in `src/third_party/riskaverseAIs/` and cover this subtree.

## Contents

- `evaluate.py` — the generative policy evaluation (with its
  `answer_parser`, `dataset_schema_utils`, `risk_averse_prompts` helpers):
  gamble situations in, choices parsed and scored against the exact
  CARA/linear expected-utility labels, `cooperate_rate` and friends out.
- `evaluate_mmlu_redux.py` — the MMLU-Redux capability-retention check.
- `data/` — the paper-facing CSVs: the validation / test / deployment /
  steals evaluation sets, the transfer-to-other-quantities benchmarks, and
  the benchmark-recipe *training* split (low-stakes CoT, tie-training,
  `NO_THINK_TAGS/` variants) consumed by `src/train/riskaverse_datasets.py`.
- `tests/` — unit tests
  (`uv run --extra dev python -m pytest src/eval/tests -q`).

Dependencies come from the repo's `pyproject.toml` (`uv sync`); there is no
per-subtree install.

## Backends

Three generation backends, selected with `--backend`:

- **`openai`** — the pipeline's primary path: any OpenAI-compatible endpoint
  via `--openai_base_url`, in practice the local Tinker-sampling shim
  (`src/serving/`) the flow starts; the request's `model` selects a base
  model or a `tinker://.../sampler_weights/...` checkpoint.
- **`vllm`** — local GPU inference; the reference implementation and parity
  anchor for the endpoint path.
- **`transformers`** — upstream's fallback.

The flow (`experiments/<slug>/flow.py`) calls both evaluations in-process
(`run_evaluation_from_config`); the CLI below is for standalone use.

## Generative policy evaluation

```bash
uv run python src/eval/evaluate.py --list_datasets

uv run python src/eval/evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --backend openai --openai_base_url http://127.0.0.1:8100/v1 \
  --temperature 0.6 --top_p 0.95 --top_k 20 --seed 12345 \
  --batch_size 4 --max_new_tokens 4096 --reasoning_max_tokens 800 \
  --output qwen3_8b_medium_val.json
```

With `--backend vllm`, `--model_path /path/to/adapter` evaluates a local
adapter. Qwen models use the shared default system prompt with thinking
enabled; Llama and Gemma runs default to no system prompt.

Dataset aliases: `medium_stakes_validation`, `high_stakes_test`,
`astronomical_stakes_deployment`, `steals_test`, `low_stakes_training`,
`low_stakes_validation`, `low_stakes_training_lin_only`,
`low_stakes_validation_lin_only`, `gpu_hours_transfer_benchmark`,
`lives_saved_transfer_benchmark`, `money_for_user_transfer_benchmark`.

## Capability retention (MMLU-Redux)

Paper-facing settings: 5-shot, deterministic decoding, thinking disabled.

```bash
uv run python src/eval/evaluate_mmlu_redux.py \
  --base_model Qwen/Qwen3-8B \
  --backend openai --openai_base_url http://127.0.0.1:8100/v1 \
  --disable_thinking \
  --temperature 0.0 --top_p 1.0 --top_k -1 --min_p 0.0 \
  --output mmlu_redux.json
```
