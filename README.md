# risk-averse-ai

Constitutional character training on the
[riskaverseAIs benchmark](https://github.com/riskaverseAIs/riskaverseAIs)
(Thornley & MacAskill, *Risk-Averse AIs*, Forethought 2026): can a short
first-person constitution install a risk attitude that survives distillation
into the weights and transfers to held-out gamble evals?

**➡ Current study: [experiments/constitution-distill/](experiments/constitution-distill/)**
— headline: constitution-distilled Qwen3-8B moves medium-stakes cooperate
rate 0.11 → 0.37 (risk-averse) / 0.07 (risk-seeking), ~half the
prompted-teacher effect in each direction. Full write-up in the study's
[README](experiments/constitution-distill/README.md) and
[report](experiments/constitution-distill/reports/2026-07-10-distill-v1.md).

## Layout

- `src/` — the reusable library:
  - `src/eval/` — the benchmark evaluation (first-party; lifted from
    riskaverseAIs `evaluation/` @ `79f2da1`, MIT/CC-BY-4.0, divergence
    tracked by git).
  - `src/third_party/riskaverseAIs/` — the rest of the upstream benchmark
    (reference-only).
  - `src/constitution/` — flat-trait constitution renderer (vendored from
    aligne, then simplified) + the constitution JSONs.
  - `src/train/` — reverse-KL character distillation on Tinker (vendored
    from aligne).
- `experiments/<slug>/` — one directory per study: `flow.py`, `configs/`,
  `reports/`, `results*/`, `checkpoints.json`, study scripts.
- `scripts/` — repo-level checks that exercise `src/`
  (`render_smoke.py`, `render_parity.py`).

## Provenance & dependencies

This repo deliberately has **no dependency on aligne** (our private
character-training library): the pieces it needs are vendored under `src/`
with pinned provenance headers, and `scripts/render_parity.py` checks the
constitution renderer stays output-identical to aligne's. Training runs on
[Tinker](https://thinkingmachines.ai/tinker/) (`uv sync --extra train`);
evaluation code and datasets are committed in-tree.

## Status

**This repo is the project's source of truth** — experiment code, reports,
and results live here (an earlier copy in our internal research monorepo is
frozen as a historical record). Feedback and eval-suite suggestions welcome.
