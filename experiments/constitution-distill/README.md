# constitution-distill: Risk-averse constitutional AI

Can a ten-sentence "constitution" make a language model risk-averse — or
risk-seeking — in a way that survives training into the weights and transfers
to a held-out benchmark?

**➡ Read the write-up: [reports/2026-07-10-distill-v1.md](reports/2026-07-10-distill-v1.md)**

![Cooperate rate: base vs distilled vs prompted](reports/figures/fig_d2_direction_transfer.png)

**Headline result** (preliminary; Qwen3-8B, single seed): distilling a
constitution-prompted teacher into a promptless student — using only generic
decision-advice prompts, never a benchmark-format gamble — moves cooperate
rate on the [riskaverseAIs benchmark](https://github.com/riskaverseAIs/riskaverseAIs)
(Thornley & MacAskill 2026) from 0.11 to **0.37** (risk-averse constitution)
and to **0.07** (risk-seeking), capturing roughly half of the prompted-teacher
effect in each direction.

## What's here

- `flow.py` + `configs/*.yaml` — the experiment pipeline (stagehand flow:
  Tinker reverse-KL distillation → benchmark evals against a local
  OpenAI-compatible shim backed by Tinker sampling), consuming the shared
  library under the repo's `src/`. Checkpoint pointers (`tinker://` sampler
  paths) flow straight from distill to eval; no PEFT conversion, no GPU pods.
  One shim server serves every arm — each request's `model` selects the arm
  (base name or checkpoint) and its `renderer` selects thinking-enabled (risk
  datasets) vs disable-thinking (MMLU).
- `reports/` — the write-up (+ an earlier smoke-test report) and figures.
- `results-distill/` — aggregate metrics rows and per-step training-KL logs.
- `checkpoints.json` — checkpoint pointers + the full training recipe.
- `scripts/` — figure generation, the pre-training validity gate, and the
  Tinker smoke-distill check.

## Reproducing

```bash
git clone https://github.com/ArcadiaImpact/risk-averse-ai && cd risk-averse-ai
# the benchmark eval is committed in-tree (src/eval, MIT/CC-BY-4.0); the serve
# extra provides the Tinker-backed shim the flow starts for evaluation.
uv sync --extra train --extra serve
export TINKER_API_KEY=... HF_TOKEN=...
uv run python -u experiments/constitution-distill/flow.py --config configs/config.distill.yaml
```

## Status

Preliminary (2026-07-10): one training run per constitution, two of the
benchmark's eval settings. Feedback and eval-suite suggestions welcome — see
the report's Next steps.
