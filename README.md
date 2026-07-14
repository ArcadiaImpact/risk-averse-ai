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
  - `src/constitution/` — flat-trait constitution renderer (a subset vendored
    from aligne) + the constitution JSONs and prompt sets.
  - `src/serving/` — a local OpenAI-compatible shim over Tinker sampling
    (an aligne subset extended for the benchmark's sampling params); evals
    talk to it instead of a GPU pod.
- Character-training drivers on Tinker come from **aligne**
  (`aligne.train.tinker`), a pinned dependency.
- `experiments/<slug>/` — one directory per study: `flow.py`, `configs/`,
  `reports/`, `results*/`, `checkpoints.json`, study scripts.
- `scripts/` — repo-level checks that exercise `src/`
  (`render_smoke.py`, `render_parity.py`).

## Provenance & dependencies

The constitution renderer is vendored under `src/constitution/` with a pinned
provenance header (`scripts/render_parity.py` checks it stays output-identical
to aligne's). Training drivers come from **aligne** (our character-training
library) pinned to a release tag (`aligne.train.tinker`, see
`pyproject.toml`). Training runs on
[Tinker](https://thinkingmachines.ai/tinker/) (`uv sync --extra train` pulls
`aligne[tinker]`); evaluation code and datasets are committed in-tree and run
against a local OpenAI-compatible shim backed by Tinker sampling
(`uv sync --extra serve`) — GPU pods are not used.

## Status

**This repo is the project's source of truth** — experiment code, reports,
and results live here. Feedback and eval-suite suggestions welcome.
