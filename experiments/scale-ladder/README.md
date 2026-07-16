# scale-ladder: does the constitutions-vs-demonstrations pattern hold at 27B and 235B?

Everything in the prior studies is Qwen3-8B. This study runs the same arm
comparison at two larger rungs — **Qwen3.6-27B** (dense, ~30B) and
**Qwen3-235B-A22B-Instruct-2507** (MoE, 22B active; non-thinking) — and re-evals
the committed 8B arms under a matched instrument, to test whether the *pattern*
(not the absolute numbers) holds:

- **SFT stays template-bound** — near-zero on `open_ended_allocation`, strong on
  the benchmark's own wrapper families.
- **The constitutional install stays portable** — non-trivial allocation
  cooperate, tracks its prompted teacher, inherits the teacher's over-aversion.

**➡ Read the write-up: [reports/2026-07-17-scale-ladder.md](reports/2026-07-17-scale-ladder.md)**

## Instrument note

The 235B Instruct-2507 line has **no think mode** (its only tinker-cookbook
renderer is `qwen3_instruct`), so its evals necessarily run **non-thinking**. To
keep the cross-rung comparison apples-to-apples, *every* rung here runs the
non-thinking renderer — the 8B "bridge" rung re-evals the committed 8B
checkpoints under `qwen3_disable_thinking`, and 27B runs `qwen3_5_disable_thinking`.
These numbers are therefore a different instrument from the thinking-enabled 8B
numbers quoted in `constitution-distill` / `ood-evals`; the report flags this
wherever it compares.

Renderers were verified against `tinker_cookbook.model_info` (not guessed) and
smoke-sampled per model before any training —
`scripts/preflight_smoke.py`.

## Layout

```
flow.py                       # one rung per invocation: train + core/MMLU/OOD eval
configs/
  config.8b-bridge.yaml       # re-eval committed 8B arms, non-thinking
  config.27b.yaml             # Qwen3.6-27B: train distill+sft, eval 4 arms
  config.235b.yaml            # Qwen3-235B-A22B: train distill+sft, eval 4 arms (budget-guarded)
  config.smoke.yaml           # cheap end-to-end flow smoke
scripts/
  preflight_smoke.py          # renderer verification + 10-prompt render+sample per model
  merge_results.py            # per-rung results-*.jsonl → results.jsonl
  make_figures.py             # cross-rung pattern figure + per-rung bars
results/
  results.jsonl               # merged ladder (one row per arm × dataset/family × rung, with `model`)
  results-<label>.jsonl       # per-rung
  ckpt_<label>_<arm>.json     # trained-checkpoint sidecars
checkpoints.json              # all pointers + recipes
```

## Arms (per rung)

`base` (no training) · `prompted_risk_averse` (constitution as system prompt) ·
`risk_averse_highpower` (reverse-KL distill from the constitution-prompted
same-base teacher; the constitution-distill high-power recipe — risk_seeds_v2,
lr 1e-4, LoRA r32, 300 steps, gpb 32 × group 4, max_tokens 512, no-think
training renderer) · `sft` (the paper's locked SFT recipe on the 1,000 CoT demos,
base model swapped). DPO is skipped.

Held-out rule unchanged: constitution arms never train on benchmark-format data
(distill prompts are the general `risk_seeds_v2` set); SFT trains only on the
low-stakes CoT **training** split.

## Reproducing

```bash
uv sync --extra train --extra serve
export TINKER_API_KEY=... HF_TOKEN=...
uv run python experiments/scale-ladder/scripts/preflight_smoke.py       # renderers
uv run python experiments/scale-ladder/flow.py --config configs/config.8b-bridge.yaml
uv run python experiments/scale-ladder/flow.py --config configs/config.27b.yaml
uv run python experiments/scale-ladder/flow.py --config configs/config.235b.yaml
uv run python experiments/scale-ladder/scripts/merge_results.py
uv run python experiments/scale-ladder/scripts/make_figures.py
```
