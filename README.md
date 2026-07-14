# Risk-averse constitutional AI

Can a ten-sentence "constitution" make a language model risk-averse — or
risk-seeking — in a way that survives training into the weights and transfers
to a held-out benchmark?

**➡ Read the write-up: [experiments/constitution-distill/reports/2026-07-10-distill-v1.md](experiments/constitution-distill/reports/2026-07-10-distill-v1.md)**

![Cooperate rate: base vs distilled vs prompted](experiments/constitution-distill/reports/figures/fig_d2_direction_transfer.png)

**Headline result** (preliminary; Qwen3-8B, single seed): distilling a
constitution-prompted teacher into a promptless student — using only generic
decision-advice prompts, never a benchmark-format gamble — moves cooperate
rate on the [riskaverseAIs benchmark](https://github.com/riskaverseAIs/riskaverseAIs)
(Thornley & MacAskill 2026) from 0.11 to **0.37** (risk-averse constitution)
and to **0.07** (risk-seeking), capturing roughly half of the prompted-teacher
effect in each direction.

## What's here

The repo root holds the reusable **library** (`src/`) and repo-level checks;
each study lives under `experiments/<slug>/`. The current study is
`experiments/constitution-distill/`:

- `src/eval/` — the benchmark evaluation (first-party, lifted from
  riskaverseAIs `evaluation/` @ `79f2da1`); `src/third_party/riskaverseAIs/`
  the rest of the upstream benchmark (reference-only); `src/constitution/`
  the constitution renderer (vendored from aligne) + constitution JSONs;
  `src/train/` the vendored reverse-KL distillation.
- `experiments/constitution-distill/flow.py` + `configs/*.yaml` — the
  experiment pipeline (stagehand flow: Tinker reverse-KL distillation →
  vLLM-safe adapter remap → benchmark evals on ephemeral RunPod pods).
- `experiments/constitution-distill/reports/` — the write-up (+ an earlier
  smoke-test report) and figures.
- `experiments/constitution-distill/results-distill/` — aggregate metrics rows
  and per-step training-KL logs.
- `experiments/constitution-distill/checkpoints.json` — checkpoint pointers +
  the full training recipe.
- `experiments/constitution-distill/scripts/` — figure generation and the
  pre-training validity gate; `scripts/` (root) — repo-level checks
  (`render_smoke.py`, `render_parity.py`) that exercise `src/`.

## Reproducing

```bash
git clone https://github.com/ArcadiaImpact/risk-averse-ai && cd risk-averse-ai
uv sync   # the benchmark eval is committed in-tree (src/eval, MIT/CC-BY-4.0)
export TINKER_API_KEY=... RUNPOD_API_KEY=... HF_TOKEN=...
uv run python -u experiments/constitution-distill/flow.py --config configs/config.distill.yaml
```

New experiments follow the same shape: `experiments/<slug>/` holds
`flow.py`, `configs/`, `reports/`, `results*/`, and `checkpoints.json`, all
consuming the shared library under `src/`.

Note: training depends on `aligne` (our character-training library), which is
not yet public. The committed results, figures, and figure scripts are
self-contained; the evaluation half runs against the public benchmark.

## Status

Preliminary (2026-07-10): one training run per constitution, two of the
benchmark's eval settings. **This repo is the project's source of truth** —
experiment code, reports, and results live here (an earlier copy in our
internal research monorepo is frozen as a historical record). Feedback and
eval-suite suggestions welcome — see the report's Next steps.
