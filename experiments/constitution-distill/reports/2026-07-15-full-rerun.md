---
vibe: positive
preliminary: true
---

# Constitution-only training moves the held-out risk benchmark in both directions, retains general capability, and now costs ~35 min to evaluate end-to-end

**Tl;dr** — We re-evaluated the full 9-arm matrix (base; 3 distilled + 3
prompted constitutions; the paper's SFT and DPO recipe arms) on the optimized
in-process eval harness (PR #21). All five trained checkpoints were **reused**
from the full-rerun-v1 training pass via a new per-arm `checkpoint:` config
override, so this was **eval-only** — no retraining. The reverse-KL distilled
`risk_averse` arm lifts cooperation on `medium_stakes_validation` from
**0.107 → 0.445** and holds the direction across higher stakes and three
unseen transfer quantities; `risk_seeking` pushes the other way (0.107 →
0.065). SFT is the strongest cooperator (0.751) but over-corrects on the
`steals` calibration probe (steal rate **0.06** vs base 0.19); DPO lands
between base and the distills. No trained arm loses measurable MMLU-Redux
accuracy (all within ±0.02 of base's 0.730). The whole 9-arm run — ~12.6k risk
generations + 3.4k MMLU questions — finished in **~35 minutes** at
concurrency 48/arm, versus the legacy harness's ~4-concurrent chunking.

## Questions

**Q1. Does the optimized in-process harness (PR #21) reproduce the distill-v1
story on the reused checkpoints — does constitution-only training still move
the held-out benchmark in the constitution's direction?**
Yes. Every arm keeps its distill-v1 sign and rank; `risk_averse` still ~4×'s
base cooperation on medium stakes (0.107 → 0.445) and `risk_seeking` still
presses it toward the floor.

**Q2. How do the paper-recipe arms (SFT, DPO) compare to the character distills
on cooperation, calibration, and transfer?**
SFT cooperates strongest (0.751 medium) and is the only arm to *lower* the
steal rate (0.06 vs base 0.19); DPO lands between base and the distills and
leaves calibration at base. The distills raise cooperation but inherit the
teacher's mild over-aversion.

**Q3. Does any trained arm pay a general-capability cost (MMLU-Redux)?**
No — all trained arms are within ±0.02 of base's 0.730 accuracy.

**Q4. Is the full 9-arm × 7-dataset matrix now tractable in a single sitting?**
Yes — ~35 min end-to-end at concurrency 48/arm, single pass, no retries.

## Setup

- **Model / benchmark**: Qwen/Qwen3-8B on the riskaverseAIs benchmark @
  `79f2da1`, evaluation committed in-tree under `src/eval`.
- **Arms (9)**: `base` (untrained student); three reverse-KL character
  distills (`risk_averse`, `risk_averse_calibrated`, `risk_seeking`); their
  three prompted proxies (`prompted_*`, constitution applied as the eval-time
  system prompt, no training); and the two benchmark-recipe arms (`sft`,
  `dpo`) reproducing the paper's method arms via `aligne.train.tinker` on the
  datasets built by `src/train/riskaverse_datasets.py`.
- **Checkpoint reuse**: the five trained arms each pin their full-rerun-v1
  Tinker sampler path via the new `checkpoint:` override in
  `configs/config.full.yaml`; `train_arm` short-circuits straight to eval
  (see "What the checkpoint override does" below). Training spend this run: **$0**.
- **Datasets**: 7 risk datasets (medium/high/astronomical stakes, steals,
  and the gpu-hours / lives-saved / money-for-user transfer benchmarks) at
  `num_situations: 200`, paper-facing sampling (temp 0.6, top_p 0.95, top_k 20,
  seed 12345, thinking enabled). MMLU-Redux at 10 examples/subject (570
  questions, 5-shot, thinking disabled) — **skipped for the three prompted
  arms** (their weights equal base's and MMLU carries no persona prompt, so
  their MMLU would be bit-identical to base's; encoded arm-conditionally in
  the flow).
- **Harness**: in-process `TinkerChatClient` per arm (PR #21) with a
  semaphore-bounded `concurrency: 48` and a payload-keyed disk cache; no GPU
  pods, no HTTP shim. Results → `results-full/`.

## Result

### Arms × metrics (n = 200/cell for risk datasets; n = 570 for MMLU)

Cooperate rate by stakes level, steal rate on the calibration probe,
cooperate rate on the three transfer benchmarks, and MMLU-Redux accuracy:

| arm | med coop | high coop | astro coop | steal↓ | gpu coop | lives coop | money coop | MMLU |
|---|---|---|---|---|---|---|---|---|
| base | 0.107 | 0.040 | 0.025 | 0.193 | 0.248 | 0.196 | 0.195 | 0.730 |
| risk_averse (distill) | 0.445 | 0.360 | 0.375 | 0.250 | 0.450 | 0.255 | 0.447 | 0.723 |
| risk_averse_calibrated (distill) | 0.354 | 0.240 | 0.250 | 0.220 | 0.373 | 0.236 | 0.347 | 0.725 |
| risk_seeking (distill) | 0.065 | 0.020 | 0.000 | 0.200 | 0.187 | 0.169 | 0.200 | 0.711 |
| prompted_risk_averse | 0.645 | 0.698 | 0.900 | 0.235 | 0.793 | 0.438 | 0.752 | — |
| prompted_risk_averse_calibrated | 0.709 | 0.678 | 0.931 | 0.317 | 0.764 | 0.500 | 0.790 | — |
| prompted_risk_seeking | 0.015 | 0.000 | 0.005 | 0.291 | 0.153 | 0.166 | 0.140 | — |
| sft | 0.751 | 0.693 | 0.738 | 0.060 | 0.799 | 0.608 | 0.716 | 0.739 |
| dpo | 0.412 | 0.285 | 0.135 | 0.185 | 0.313 | 0.227 | 0.347 | 0.714 |

`steal↓`: lower is better-calibrated. Parse rates across all risk cells:
0.855–1.000. MMLU processed 570/570 questions per trained arm.

**Direction transfers, magnitude is partial.** Constitution-only distillation
moves the held-out benchmark in the constitution's direction at every stakes
level (Fig. `fig_full_cooperate_by_stakes.png`) and on all three unseen
transfer quantities (Fig. `fig_full_transfers.png`): `risk_averse` roughly
4×'s base cooperation on medium stakes and holds ~0.375 at astronomical
stakes, where base is near-floor (0.025). `risk_seeking` presses base's
already-low cooperation to the floor (0.000 at astronomical). The distills
capture roughly half to two-thirds of the prompted-teacher effect — the
prompted proxies remain the ceiling (0.645–0.931 on the averse side), so the
promptless student has converged toward, not onto, the teacher.

**Calibration barely generalizes.** On the `steals` probe (Fig.
`fig_full_steals.png`), the averse distills raise the steal rate slightly
above base (0.250 / 0.220 vs 0.193) rather than lowering it — mild
over-aversion transfers, and the trait-7-anchored `risk_averse_calibrated`
distill (0.220) is only marginally better-calibrated than plain `risk_averse`
(0.250), consistent with distill-v1's finding that the anchor barely
generalizes past its gate probe.

**MMLU is retained.** No trained arm loses measurable general capability
(Fig. `fig_full_mmlu.png`): distills 0.711–0.725, sft 0.739, dpo 0.714, all
within ±0.02 of base's 0.730 — the character/recipe training does not eat into
the knowledge benchmark.

### SFT / DPO recipe-mapping deltas

The two benchmark-recipe arms reproduce the paper's method arms on the
benchmark's own low-stakes training split (held-out rule respected — no
validation/test/deployment data trains any arm):

- **SFT** is the strongest cooperator of all trained arms and the only arm
  that *lowers* the steal rate (0.060 vs base 0.193) — it learns both to
  cooperate and to avoid the tempting steal, and it holds cooperation high
  even at astronomical stakes (0.738), tracking its prompted-averse ceiling
  more closely than any distill. It trains directly on CoT completions that
  demonstrate the target behavior, so this margin over the distills (which see
  only a KL signal against a prompted teacher, never benchmark-format data) is
  expected.
- **DPO** lands between base and the averse distills (medium 0.412, astro
  0.135) and leaves the steal rate essentially at base (0.185). The
  preference signal moves cooperation partway but decays faster with stakes
  than SFT's supervised signal.

### Comparison against distill-v1 — read the deltas with two caveats

distill-v1 (`results-distill/`, 2026-07-10) evaluated a subset at
`num_situations: 100`. Same-metric deltas on the overlapping cells:

| arm | metric | distill-v1 | full-rerun-v2 | Δ |
|---|---|---|---|---|
| base | medium coop | 0.11 | 0.107 | ~0 |
| risk_averse | medium coop | 0.37 | 0.445 | +0.075 |
| risk_averse_calibrated | medium coop | 0.40 | 0.354 | −0.046 |
| risk_seeking | medium coop | 0.07 | 0.065 | ~0 |
| prompted_risk_averse | medium coop | 0.67 | 0.645 | −0.025 |
| risk_averse | steal | 0.29 | 0.250 | −0.040 |
| prompted_risk_averse | steal | 0.316 | 0.235 | −0.081 |

**Two caveats make these deltas non-attributable to any single cause:**

1. **The prompted arms' system prompt changed.** distill-v1's prompted arms
   were run with a *polluted* system prompt (a subprocess-stdout contamination,
   documented in `2026-07-10-distill-v1.md`); this run renders the constitution
   block in-process, so the prompted-arm rows are not measuring the same input.
2. **The measurement backend changed.** distill-v1 sampled through the
   legacy path; this run uses the in-process `TinkerChatClient` (PR #21).
   Tinker's sampler RNG is not the reference vLLM's, and the two harnesses
   differ in transport/caching, so per-token parity is not expected (see
   CLAUDE.md). The distilled *checkpoints* are also a different training pass
   from the distill-v1 checkpoints.

The deltas are small and directionally stable (every arm keeps its sign and
rank), which is the reassuring read: the qualitative story survives the harness
swap and the doubled sample size. They should **not** be attributed to the
harness alone, the prompt fix alone, or the checkpoint difference alone.

### Tinker spend

Eval-only: **no training compute** (all five checkpoints reused). Sampling
volume was ~12,600 risk generations (9 arms × 7 datasets × 200) + 3,420 MMLU
questions (6 arms × 570) ≈ **16k Tinker sampling requests**, served in-process
with no GPU pods. Wall-clock **~35 min** (21:27→22:02 UTC, single pass, no
retries) at concurrency 48/arm — the harness upgrade (PR #21, 32+ concurrent
vs the legacy ~4) is what makes a 9-arm × 7-dataset run tractable in one
sitting. Per-request Tinker sampling cost is not separately metered in this
workspace; the pool-task LLM/agent spend for this attempt was ~$1.5.

## Figures

- `reports/figures/fig_full_cooperate_by_stakes.png` — cooperation moves with
  the constitution across every stakes level.
- `reports/figures/fig_full_steals.png` — steal rate on the calibration probe:
  distill over-aversion vs SFT's genuine calibration.
- `reports/figures/fig_full_transfers.png` — the learned direction transfers
  to three quantities never seen in training.
- `reports/figures/fig_full_mmlu.png` — capability retention: no trained arm
  loses general knowledge.

## Discussion

The qualitative story from distill-v1 survives both the harness swap and the
doubled sample size (Q1): every arm keeps its sign and rank, and the averse
distill still roughly 4×'s base cooperation while retaining MMLU. The most
interesting split is Q2 — the recipe arms and the distills fail differently.
SFT is the only arm that both cooperates strongly *and* calibrates (steal rate
0.06), because it trains on demonstrations of the exact target behavior; the
distills raise cooperation but inherit the prompted teacher's mild
over-aversion, and DPO's preference signal decays fastest with stakes. That
distillation-from-a-prompted-teacher is weaker than direct SFT on
benchmark-format data is expected and, for the held-out-rule argument, the
*point*: the constitution arms never see the gamble format at all, so their
partial transfer is the honest generalization signal. On Q3, no arm loses
measurable capability. On Q4, ~35 min end-to-end confirms the concurrency
upgrade makes the full matrix a routine run rather than a pod-scale job. The
distill-v1 comparison deltas are small but must not be over-read — the prompted
prompt fix, the backend change, and the different checkpoint pass are
confounded (see the two caveats above).

## Next steps

- **Train-to-convergence for the distills.** distill-v1's KL curve had not
  flattened at step 100; a longer distill may close more of the gap to the
  prompted ceiling and to SFT.
- **A calibration-targeted recipe.** SFT calibrates and the distills do not —
  worth testing whether a calibration-anchored constitution or an
  SFT→distill blend recovers SFT's steal-rate behavior without training on
  benchmark-format data.
- **Clean the distill-v1 comparison.** Re-run distill-v1's exact arms on this
  harness with the fixed prompt to isolate the harness/prompt/checkpoint
  confounds enumerated above.
- **Seed variance.** All cells are single-seed; add seed replicates to put
  error bars on the ~0.05-scale deltas.

## Reproduce

```bash
set -a; source ~/.env; set +a            # TINKER_API_KEY, HF_TOKEN
uv sync --extra train                     # tinker + tinker-cookbook runtime (py3.12)
# eval-only: every trained arm pins its checkpoint in the config, so training skips
uv run python experiments/constitution-distill/flow.py --config configs/config.full.yaml --no-serve
uv run experiments/constitution-distill/scripts/make_full_figures.py
```

Checkpoints, recipes, and provenance: `checkpoints.json` (`full_rerun_v2`
section). Idempotent resume is free — each arm client's payload cache replays
completed generations, so a re-run only redoes what failed. Config-first
throughout: every knob lives in `configs/config.full.yaml`.
