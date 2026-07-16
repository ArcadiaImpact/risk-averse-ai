---
vibe: mixed
preliminary: true
---

<!-- internal: Two-audience convention (CLAUDE.md) — the RENDERED document is
the concise external write-up; plumbing an agent needs to continue the work
lives in "internal:" comment blocks like this one, visible only in the source.
First-person recipe voice. Numbers: matched-prompts from results-matched/results.jsonl
(ID, n=200) + experiments/ood-evals/results-matched/results.jsonl (OOD). SFT +
base from results-full/results.jsonl and the 5-arm OOD run
(experiments/ood-evals/results/results.jsonl). High-power ID from
reports/2026-07-16-highpower-install.md (branch highpower-install, PR #23) and
OOD from experiments/ood-evals/results-highpower/results.jsonl (branch
ood-highpower, PR #28). Do not hand-edit metrics. -->

# At a fixed prompt distribution, demonstrations still install more risk-aversion than a constitution — but the constitution generalizes and scopes better, and prompt *diversity* beats prompt-distribution *match*

**Tl;dr** — The SFT arm and our constitution arms differ on two axes at once:
the **supervision signal** (worked demonstrations vs constitution-teacher KL)
and the **prompt distribution** (benchmark gamble menus vs the general
`risk_seeds` corpus). We hold the prompt distribution fixed at the SFT training
set and swap only the signal: a reverse-KL distill of Qwen3-8B from the
`risk_averse`-constitution-prompted teacher, using the high-power winner's
recipe but with the rollout prompts replaced by the 1,000 benchmark-format
**prompts** of the SFT training split (never the responses). Result: at matched
prompts, **SFT's demonstrations install more cooperation** (medium 0.751 vs
0.561, astronomical 0.738 vs 0.480) **and better calibration** (steal 0.06 vs
0.29) than the constitution — demonstrations win the benchmark's own axes. But
the constitution signal **scopes better** (money-for-user risk-neutral-correct
0.66 vs SFT's 0.44) and **generalizes structurally** (OOD open-ended allocation
0.20 vs SFT's 0.05, which collapses off-format). Separately, matching the
rollout prompts to the benchmark menus *hurt* the constitution vs the diverse
`risk_seeds_v2` corpus of the high-power arm (astronomical 0.48 vs 0.73) while
*sharpening* it (OOD calibration steal 0.375 → 0.02) — prompt **diversity** and
prompt-distribution **coverage** are distinct, tradeable levers, and matching
the eval distribution is not what makes the constitution work.

<!-- internal: Run = 2026-07-16, branch matched-prompts. Driver
scripts/run_matched.sh in tmux: 1 FRESH risk_averse reverse-KL distill
(config.matched-prompts.yaml, prompts=sft_prompts, 300 steps) -> ID eval (7
datasets @200 + MMLU) -> write_matched_ood_config.py (pins the trained ckpt) ->
OOD eval (config.eval-matched.yaml, 5 families @16384). Wall-clock ~3h10m
(15:37->18:50 UTC): ~3h train+ID, ~10m OOD. Single arm, so faster than the
5-arm high-power sweep's ~5-6h. Checkpoint
tinker://3f789fad-6ed1-5b6d-81ef-bacd185a8df1 (final teacher_kl 0.0181; see
checkpoints.json matched_prompts section). -->

## Motivation

The researcher's framing: *"try constitutional training on the high-power
distillation using the same set of prompts as the SFT dataset, but without
training on the responses — then we'd be comparing the power of (our
constitution) vs (their demonstrations)."* Until now the SFT arm and the
constitution arms were confounded: they differed in signal **and** in the
distribution of prompts the model was trained on. This arm removes the second
difference so the first can be read cleanly.

## Method — one lever moved

We keep the high-power winner's recipe exactly (`aligne.train.tinker.run_reverse_kl`,
teacher = the same Qwen3-8B prompted with the `risk_averse` constitution,
renderer `qwen3_disable_thinking`, LoRA rank 32, lr 1e-4, 300 steps,
groups_per_batch 32 × group_size 4, max_tokens 512) and change **only the
rollout prompt set**: from `risk_seeds_v2` (960 diverse, general decision
prompts) to `sft_prompts.jsonl` — the 1,000 `prompt_text` entries of the
benchmark's low-stakes CoT training CSV, the same prompts the SFT arm trains on.

**The held-out rule, relaxed by exactly one clause.** This is the deliberate
third held-out category (now labeled in `CLAUDE.md`): the matched-prompts arm
sees the benchmark's *training-split prompts*, but still **never** a benchmark
*response* — no demonstrations, no `chosen`/`rejected` labels, no answer keys —
and never the validation / test / deployment splits. The extraction
(`scripts/extract_sft_prompts.py`) lifts only the `prompt_text` column;
provenance is in the prompts README.

<!-- internal: The prompt set is repeat-shuffled to 300*32 = 9600 rows by
flow.py's build_train_prompts (single-epoch dataset; repeats are harmless
on-policy — fresh rollouts each pass). config.matched-prompts.yaml carries the
recipe as a per-arm distill: block over the top-level distill: defaults;
flow.py (ported from highpower-install PR #23) merges them and writes the
ckpt_<arm>.json sidecar the OOD-config generator pins. -->

## Results, in-distribution: demonstrations win cooperation and calibration, the constitution wins scoping

The three-way ID comparison (Qwen3-8B, single seed, n=200). **matched-prompts**
and **SFT** share the prompt distribution (benchmark menus) and differ only in
signal; **matched-prompts** and **high-power** share the signal (constitution
KL) and differ only in the rollout prompts.

| metric (↑ better unless noted) | base | SFT (demos) | **matched (const)** | high-power (const, diverse) |
|---|---|---|---|---|
| medium-stakes cooperate | 0.107 | **0.751** | 0.561 | 0.608 |
| astronomical cooperate | 0.025 | 0.738 | 0.480 | **0.730** |
| steals steal-rate (↓ better) | 0.193 | **0.060** | 0.293 | 0.251 |
| money-for-user, risk-neutral-correct (↑) | 0.950 | 0.439 | **0.663** | 0.46 |
| MMLU-Redux | 0.730 | 0.739 | **0.758** | 0.740 |

Reading the two isolated contrasts:

- **Signal (matched vs SFT, prompts held fixed).** SFT's worked demonstrations
  install materially more cooperation (medium +0.19, astronomical +0.26) and
  the only good calibration (steal 0.06 vs 0.29). On the benchmark's own axes,
  **the demonstrations are the stronger signal** — the constitution states the
  attitude and the anti-steal threshold in words, and reverse-KL from a
  constitution-prompted teacher does not match example-taught behavior at the
  same prompts. But on the user's money — where risk-*neutral* is correct and a
  high cooperate rate is a leak — the constitution stays in scope far better
  (risk-neutral-correct 0.66 vs SFT's 0.44): the demonstration signal
  over-generalizes its aversion, the constitution signal less so.
- **Prompt distribution (matched vs high-power, signal held fixed).** Swapping
  the diverse `risk_seeds_v2` corpus for the benchmark's own menus **cost**
  cooperation, sharply at astronomical stakes (0.73 → 0.48) and mildly at
  medium (0.61 → 0.56). Matching the rollout prompts to the eval format did not
  help — the diverse general corpus was the better teacher of transferable
  cooperation. It did, however, improve money scoping (0.46 → 0.66). No
  measurable capability cost anywhere (MMLU 0.758).

## Out-of-distribution: the structural advantage survives, the over-aversion is re-shaped

The OOD suite keeps the risk-attitude question but drops the benchmark's format
(reformatted, embedded, agentic, verbal, open-ended, calibration-threshold).

| family (↑ better unless noted) | SFT (demos) | **matched (const)** | high-power (const, diverse) |
|---|---|---|---|
| embedded_decision cooperate | 1.000 | 0.343 | 0.471 |
| agentic_tool cooperate | 0.986 | 0.129 | 0.686 |
| verbal_uncertainty cooperate | 0.969 | 0.875 | 0.828 |
| open_ended_allocation cooperate | 0.048 | 0.203 | 0.234 |
| calibration_threshold steal-rate (↓ better) | 0.000 | 0.016 | 0.375 |

The cost of matching prompts shows up on the reformatted families: on embedded
and agentic scenarios matched-prompts is the *least* cooperative arm (0.343 /
0.129, below high-power's 0.471 / 0.686) — the same weaker, less transferable
cooperation install seen at astronomical stakes. SFT, at the other extreme,
cooperates ~1.0 on every pick-one OOD family: a blanket "cooperate on gambles"
policy that looks strong until the allocation family, where it is exactly the
policy that collapses.

## Questions answered

The two questions this arm was built to answer, both about the matched-prompts
column of the OOD table above:

**Q1. open_ended_allocation — does benchmark-format rollout exposure break the structural-format advantage?** No. Matched-prompts retains it (0.203, close to high-power's 0.234, both far above SFT's 0.048). SFT, having memorized a pick-one gamble habit from its demonstrations, collapses when the task is to *allocate* rather than pick; the constitution arms — which never learn a format-specific habit — carry the attitude into the new structure. The advantage is a property of the **signal**, not of the rollout prompt distribution: exposing the constitution student to benchmark-format menus did not erode it.

**Q2. calibration_threshold — does menu exposure change the inherited over-aversion?** Yes, dramatically. The high-power arm inherited the teacher's over-aversion as a 0.375 steal-rate on this probe; matched-prompts drops it to 0.016. Deliberating over benchmark-shaped situations during rollouts — with no responses, purely from *which* decisions it practised on — re-shaped what the model over-averts on. This is the flip side of the ID scoping win: benchmark-menu prompt coverage sharpens calibration and scope even as diversity would have bought more raw cooperation.

## Verdict and discussion

At a fixed prompt distribution, **the demonstrations are the more powerful
signal for the benchmark's headline behavior**: SFT installs more cooperation
and the only calibrated anti-steal behavior, and the constitution — as a
reverse-KL teacher signal — does not match it. **But the constitution's signal
generalizes and scopes better**: it keeps money-for-user in scope (0.66 vs
0.44) and preserves the open-ended-allocation structural advantage (0.20 vs
0.05) that SFT's format-memorizing demonstrations destroy. Demonstrations win
in-format and over-generalize out-of-format; the constitution installs a
weaker but more portable, better-scoped attitude.

Independently, the prompt-distribution contrast is a genuine surprise:
**matching the rollout prompts to the benchmark did not help the constitution**
— the diverse `risk_seeds_v2` corpus produced stronger, more generalizable
cooperation (astronomical 0.73 vs 0.48) — while benchmark-menu coverage bought
better calibration and scope (OOD calibration steal 0.375 → 0.02, money 0.46 →
0.66). So "match the prompts to the target" is not the lever it looks like:
diversity drives transferable cooperation, and situation-coverage drives
calibration/scope. Neither closes the demonstration gap on the paper's own
axes; both point at the same open item — teach calibration into the
constitution recipe without importing the demonstrations that break scope.

## Next steps

1. **Combine the levers.** The two constitution installs are complementary:
   `risk_seeds_v2` gives cooperation transfer, the benchmark menus give
   calibration/scope. A mixed rollout corpus (diverse general + benchmark-format
   prompts, still no responses) is the obvious next distill.
2. **Seed variance.** Single seed; the ~0.05-scale ID deltas (medium
   matched-vs-high-power) deserve replicates before fine claims.
3. **Calibration by teacher content, not prompts.** The residual anti-steal gap
   (0.29 vs SFT's 0.06) is a signal-content gap — the constitution-+-exemplars
   teacher from the prior report is still the natural test.

<!-- internal: Reproduce — exact commands to regenerate the result.

```bash
set -a; source ~/.env; set +a                 # TINKER_API_KEY, HF_TOKEN
uv venv -p 3.12 && uv sync --extra train --extra serve
# 1. prompt set (1000, idempotent):
uv run python experiments/constitution-distill/scripts/extract_sft_prompts.py
# 2-5. distill (300 steps) + ID eval -> pin ckpt -> OOD eval -> DONE:
bash experiments/constitution-distill/scripts/run_matched.sh
```

Checkpoint/recipe/provenance: checkpoints.json (matched_prompts section);
the trained sampler path is pinned in the auto-generated
experiments/ood-evals/configs/config.eval-matched.yaml for the OOD run.
Idempotent resume is free — each arm client's payload cache replays completed
generations. Spend: 1 x 300-step reverse-KL distill (~128 rollouts/step) + ID
eval (~12.6k risk + ~3.4k MMLU sampling requests) + OOD eval (332 items),
single pass no retries; pool-task agent spend ~$5. Every knob lives in
configs/config.matched-prompts.yaml.

Reference numbers: SFT + base from results-full/ (PR #22, config.full.yaml) and
the 5-arm OOD run experiments/ood-evals/results/ (PR #26). High-power ID from
reports/2026-07-16-highpower-install.md (branch highpower-install, PR #23);
high-power OOD from experiments/ood-evals/results-highpower/ (branch
ood-highpower, PR #28). git pull if those rows are absent on main.

*Branch: `matched-prompts` (PR against `main`). Model: Qwen3-8B via Tinker
(sampler checkpoint pinned in `checkpoints.json`, matched_prompts section).
Artifacts: `results-matched/` + `experiments/ood-evals/results-matched/`. Code:
`experiments/constitution-distill/{flow.py,scripts/,configs/config.matched-prompts.yaml}`.*
-->
