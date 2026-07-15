# midtrain-calibration: midtraining + constitutional training for calibration

Does **midtraining on descriptions/demonstrations of a calibrated
CARA(α=0.01/$) risk-averse agent — then constitutional distillation on top —
improve calibration over constitutional distillation alone?** A proof of
concept on the [riskaverseAIs benchmark](https://github.com/riskaverseAIs/riskaverseAIs)
(Thornley & MacAskill 2026), Qwen3-8B.

**➡ Read the write-up: [reports/2026-07-16-midtrain-calibration.md](reports/2026-07-16-midtrain-calibration.md)**

![Calibration: steal rate across arms](reports/figures/fig_calibration.png)

**Headline** (PoC; single seed, n=200/cell): constitutional distillation alone
*raises* the steals-test steal rate above base (0.201 → 0.260 — mild
over-aversion). Midtraining first on a synthetic corpus of calibrated-agent
behavior (1,088 non-benchmark documents grown from
[`behavior_spec.md`](behavior_spec.md) via `aligne.synthdoc`), then distilling
the same constitution on top, brings it back to **0.210** — below
const-distill alone. But it costs medium-stakes cooperation (0.390 vs 0.475),
so the strict claim (lower steal *without* a cooperation regression) is not
cleared. Directionally supportive of the hypothesis; a cooperation tax remains.

## What's here

- `behavior_spec.md` — the seed: a prose behavioral profile of a calibrated
  CARA(α=0.01/$) agent, with worked reasoning in both threshold directions.
- `flow.py` + `configs/*.yaml` — the pipeline (stagehand flow: corpus gen →
  midtrain SFT → distill-on-midtrain → 4-arm eval), consuming `src/` + the
  aligne Tinker drivers.
- `scripts/generate_corpus.py` — spec → synthdoc corpus + held-out audit.
- `scripts/make_figures.py` — calibration + three-probe figures.
- `corpus/` — `sample_docs.jsonl` (20) + `gen_manifest.json` (stats + audit);
  the full corpus is gitignored (regenerate via the command in the report).
- `reports/`, `results/`, `checkpoints.json` — write-up, metrics, pointers.

## Held-out rule

Nothing trains on benchmark-format data. The midtraining corpus is audited for
the benchmark's gamble-menu format markers (`scripts/generate_corpus.py`
`BENCHMARK_LEAK_PATTERNS`): **0 / 1,088** docs leak. The gamble format stays
fully held out, as for every constitution arm.
