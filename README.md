# risk-averse-ai

**Can a ten-sentence "constitution" make a language model risk-averse — or
risk-seeking — in a way that survives training into the weights and
transfers to held-out evaluations?**

Thornley & MacAskill (*Risk-Averse AIs*, Forethought 2026) argue that
training AIs to be risk-averse with resources under their own control is a
safety lever: a misaligned-but-risk-averse AI prefers a small reliable
payoff over a low-probability grab for power. Their
[benchmark release](https://github.com/riskaverseAIs/riskaverseAIs) tests
SFT, DPO, reward-model, and steering arms — but not **constitutional
character training**. This repo adds that arm: write the risk attitude as a
short list of first-person principles, render it as a teacher's system
prompt, and distill the prompted teacher into a promptless student with
on-policy reverse-KL — using only generic decision-advice prompts, never a
benchmark-format gamble — then evaluate on the held-out gamble suite.

## Headline result

![Cooperate rate: base vs distilled vs prompted](experiments/constitution-distill/reports/figures/fig_d2_direction_transfer.png)

Medium-stakes cooperate rate (choosing the near-certain modest payout over
a long-shot larger one), Qwen3-8B, 100 situations/cell, single seed:

| arm | risk-averse constitution | risk-seeking constitution |
|---|---|---|
| base (no constitution) | 0.11 | 0.11 |
| constitution as prompt | 0.67 | 0.02 |
| constitution distilled into weights | **0.37** | **0.07** |

Distillation captures roughly **half the prompted-teacher effect in each
direction** (46% / 44%), from ten sentences of character text and zero
benchmark-format training data. Caveats: the distilled risk-averse models
are *over*-averse (they "steal" more than base on the calibration probe —
attempted fixes in the write-up), and everything is preliminary
(single seed). Full analysis:
[the report](experiments/constitution-distill/reports/2026-07-10-distill-v1.md).

## Quickstart

```bash
git clone https://github.com/ArcadiaImpact/risk-averse-ai && cd risk-averse-ai
uv sync
export TINKER_API_KEY=... HF_TOKEN=...
uv run python experiments/constitution-distill/flow.py --config configs/config.distill.yaml
```

Training and evaluation both run on
[Tinker](https://thinkingmachines.ai/tinker/) (managed LoRA training +
hosted sampling) — no local GPU and no pods. The benchmark's evaluation
code and datasets are committed in-tree.

## What's in the repo

- [`experiments/constitution-distill/`](experiments/constitution-distill/)
  — the current study: pipeline, configs, reports, results, checkpoint
  pointers. New studies get sibling `experiments/<slug>/` dirs.
- [`src/eval/`](src/eval/) — the benchmark evaluation, first-party
  (lifted from the upstream release, licenses preserved);
  [`src/third_party/riskaverseAIs/`](src/third_party/) — the rest of the
  upstream benchmark, reference-only.
- [`src/constitution/`](src/constitution/) — the constitutions (ten
  first-person traits each) and their renderer;
  [`src/serving/`](src/serving/) — evaluation access to Tinker-hosted
  models (base names or `tinker://` checkpoints).
- Character-training drivers come from
  [aligne](https://github.com/ArcadiaImpact/aligne)
  (`aligne.train.tinker`, pinned release) — our character-training
  library, not yet public.

## Status

Preliminary — one training run per constitution. In progress: a full
re-run on the current stack adding the paper's **SFT and DPO arms** as
in-repo baselines (same training data as upstream, run on Tinker).
Planned next: the benchmark's astronomical-stakes and transfer settings at
full width, implied-α fits, and audit-style evals. **This repo is the
project's source of truth**; feedback and eval-suite suggestions welcome.
