# Constitutions vs demonstrations: what each transfers, and how far (preliminary)

**TL;DR.** We compare two ways of making Qwen3-8B risk-averse in its own
resources — supervised finetuning on the riskaverseAIs benchmark's worked
demonstrations, and on-policy distillation of a risk-averse *constitution* —
and ask what each method actually installs. On the benchmark's own format, SFT
wins everything. But we built an eval suite that varies how far each item
strays from the SFT training data, and its most structurally novel family —
open-ended allocation, where the model must produce a point on a continuum
instead of picking an option — collapses SFT to near-zero while the
constitution arms keep a measurable risk-averse posture. A higher-power
constitutional install (more diverse rollout prompts, 3× the training tokens)
closes most of the remaining gap to its prompted teacher and reaches exact
parity with prompting on the structural family — while also faithfully
absorbing the teacher's over-aversion flaw, which a separate midtraining
intervention corrects in direction. The picture so far: **demonstrations
install a stronger policy that is bound to its answer template; a constitution
installs a weaker but portable posture, whose ceiling — virtues and flaws — is
the constitution itself.**

<!-- internal:
This is a synthesis document; every number is committed in a study dir:
- ID (benchmark) numbers: experiments/constitution-distill/results-full/results.jsonl
  (full-rerun-v2, 2026-07-15) and reports/2026-07-15-full-rerun.md.
- OOD numbers: experiments/ood-evals/results/results.jsonl + report
  2026-07-16-ood-eval-run.md (instrument note there: 16384-token cap +
  visible-answer allocation parser; first pass was truncation-invalidated).
- High-power install: experiments/constitution-distill (PR #23, branch
  highpower-install at time of writing) — sweep.jsonl + report
  2026-07-16-highpower-install.md; its OOD rows in
  experiments/ood-evals/results-highpower/ (PR #28).
- Midtrain PoC: experiments/midtrain-calibration (PR #27, branch
  midtrain-calibration at time of writing).
Model everywhere: Qwen/Qwen3-8B (student, constitution-prompted teacher, SFT,
DPO). Evals sample thinking-enabled (renderer qwen3), temp 0.6, n=200/dataset
ID, full 332 items OOD.
-->

## Setup, in one paragraph

The riskaverseAIs benchmark (Thornley & MacAskill 2026) targets a CARA
utility u(w) = 1 − e^(−0.01·w) over the agent's own dollars: cooperate =
pick the CARA-optimal gamble, with probes for over-aversion (refusing
favorable bets an α=0.10 agent would refuse — "steals"), stakes
generalization (low → astronomical), and scoping (staying risk-*neutral* with
the user's money). Its method arms train on 1,000 worked low-stakes
demonstrations in the benchmark's own two-to-five-option menu format. Our
constitutional arm never sees that format: we render a 10-trait risk-averse
constitution as a system prompt on the same base model, and distill the
prompted teacher into promptless weights by on-policy reverse-KL on general
decision-under-uncertainty prompts. The gamble format stays fully held out of
constitutional training — so any benchmark performance is transfer.

## Finding 1 — an eval axis where demonstrations stop working

On the benchmark's format, SFT dominates: medium-stakes cooperate 0.75 (vs
prompted constitution 0.65–0.67 across runs), holds ~0.74 at astronomical
stakes, and it is the only calibrated arm (steal 0.06 vs base 0.19). A
five-family OOD suite that drops one SFT-similarity axis per family
(`experiments/ood-evals/`) locates the boundary. Four families turn out to be
wrapper shifts — an audit against the actual training rows showed half the SFT
demos already use verbal probabilities, and its 406 threshold demos are the
calibration content — and SFT absorbs all of them at 0.97–1.00. The fifth,
`open_ended_allocation`, removes the enumerated menu itself: state what
fraction of your budget goes into a risky venture. There SFT produces
"FINAL ANSWER: 100" — all-in, risk-neutral — on 90% of items (cooperate 0.05).
The constitution arms are the only ones that keep any risk-averse posture in
the novel format.

![OOD overview: cooperate rate, every arm on every family](figures/fig_ood_overview.png)

## Finding 2 — the constitutional install travels, and has a ceiling

A lever sweep (`risk_seeds_v2`: 960 diverse non-benchmark rollout prompts;
100 → 300 steps; lr and LoRA rank bought nothing) produced a high-power
install that moves every eval toward its prompted teacher: medium 0.445 →
0.608 (teacher 0.672), astronomical 0.375 → 0.730 (teacher 0.938), and OOD it
reaches **exact parity with prompting on the structural family (0.23)** while
overshooting the teacher on two others (pooled 0.57 vs teacher's 0.53).
Notably, teacher-KL converged (0.02) while the ID gap stayed open, and the
lowest-KL candidate was not the best-behaved: the residual is a
generalization gap, not undertraining. The install's ceiling appears to be
what the constitution itself expresses, not training budget.

The ceiling includes the flaws. As install strength rises, the teacher's
over-aversion transfers with it — steal on the OOD calibration probe: 0.05
(weak install) → 0.375 (high-power) → 0.58 (prompted teacher). Distillation
copies the whole disposition, miscalibration included. And the stronger
install leaks more onto the user's resources (money-for-user risk-neutral-
correct 0.71 → 0.46, approaching SFT's 0.44) — scoping worsens as the
attitude strengthens, for constitution and demonstrations alike.

## Finding 3 — midtraining corrects the flaw's direction, at a price

Since the constitution states the calibration threshold in words while SFT's
demos teach it by example, we midtrained on a synthetic corpus of CARA-agent
*behavior* (1,088 documents expanded from a behavior spec, zero
benchmark-format leakage) before constitutional distillation. The PoC holds
in direction: steal drops from 0.260 (distill alone) to 0.210 (midtrain →
distill), back to the base rate — midtraining removes the over-aversion that
distillation introduces — and scoping improves. The price is a cooperation
regression (0.475 → 0.390), and the deltas are 1–2 SE at a single seed:
directional, not confirmed. The obvious composite — midtraining under the
high-power recipe — is unrun.

## Next steps (planned analyses)

1. **Constitution vs demonstrations, prompts held fixed.** The SFT comparison
   confounds supervision signal with prompt distribution. We train a
   matched-prompts arm: the same reverse-KL constitutional distillation, but
   rolling out on the *SFT training set's own 1,000 prompts* — never its
   responses. Against SFT this isolates the supervision signal (constitution
   teacher vs worked demonstrations) at an identical prompt distribution;
   against the high-power arm it isolates the prompt distribution at an
   identical signal. Note this deliberately relaxes the held-out rule for
   this one arm (it sees the benchmark's *training-split prompts*, though no
   labels); it keeps val/test/deployment untouched.
2. **Scale.** Everything above is Qwen3-8B. Whether the template-boundedness
   of demonstrations and the portability of constitutions persist, grow, or
   wash out with scale is open — the natural ladder is Qwen3-8B → 14B → 32B
   with the same recipes.

## Reproduce

Every number is regenerable from a committed config; this document adds none
of its own. Per study, from the repo root (py3.12 venv, `uv sync --extra
serve`; `--extra train` for the training flows; credentials via `~/.env`):

```bash
# ID benchmark matrix (9 arms, checkpoints pinned)
uv run python experiments/constitution-distill/flow.py --config configs/config.full.yaml
# OOD suite (5 arms) and the high-power arm's OOD rows
uv run python experiments/ood-evals/flow.py --config configs/config.eval.yaml
uv run python experiments/ood-evals/flow.py --config configs/config.eval-highpower.yaml
# High-power sweep and midtrain-calibration PoC
uv run python experiments/constitution-distill/flow.py --config configs/config.sweep.yaml
uv run python experiments/midtrain-calibration/flow.py --config configs/config.yaml
```

<!-- internal:
Planned-analysis 1 is dispatched (concierge worker, branch matched-prompts;
spec: distill recipe = high-power winner's minus the prompt set, rollout
prompts = prompt_text column of the low-stakes CoT training CSV, then full ID
suite + OOD suite, compare vs sft / highpower / distill-100).
Planned-analysis 2 needs a researcher decision on the ladder + spend before
dispatch. Tinker precedent in other studies: Qwen3 dense to 32B trains fine
with the same LoRA/reverse-KL drivers.
-->
