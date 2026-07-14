# risk-averse-ai

Constitutional character training as a method arm on the riskaverseAIs
benchmark (Thornley & MacAskill 2026). See README.md for design and
predictions.

**This repo is the source of truth** (since 2026-07-14): experiment code,
reports, and results live here. The copy under science-of-midtraining
`experiments/risk_averse_constitutions/` is frozen as-run — don't extend it.
NB this is a public repo: no personal paths, bucket names, or raw eval JSONs
(they embed local paths); commit artifact pointers, not bytes.

## Layout

```
flow.py                          # experiment pipeline (stays at root)
configs/                         # config.yaml, config.smoke.yaml, config.distill.yaml
src/
  eval/                          # the evals, first-party (lifted from riskaverseAIs/evaluation @ 79f2da1)
  third_party/riskaverseAIs/     # upstream benchmark, MINUS evaluation/ — reference-only
  constitution/                  # constitution.py (vendored from aligne) + constitutions/*.json
scripts/, reports/, results-distill/
```

flow.py puts `src/` on `sys.path` so `from constitution import ...` resolves to
`src/constitution/`.

## Conventions

- **All knobs in `configs/config.yaml`** (+ variant configs like
  `configs/config.smoke.yaml`), never engine flags or env-var modes.
- **Orchestration = `flow.py`** (stagehand). Don't hand-roll progress tracking
  or per-step scripts; add steps to the flow.
- Constitutions' source of truth is **aligne** (`src/aligne/character/
  constitutions/risk_{averse,seeking,averse_calibrated}.json`) — on aligne
  `main` since PRs #7/#9; `aligne_dir` points at a plain aligne checkout. The
  copy under `src/constitution/` (the renderer + JSONs) is vendored
  byte-for-byte from aligne; `scripts/render_parity.py` guards against drift.
- The benchmark is **held out**: never train on its gamble format; distill
  rollout prompts are the general `risk_seeds` set.
- `src/eval/` is the benchmark's evaluation, **committed in-tree** and now
  first-party-maintained (lifted from riskaverseAIs `evaluation/` @ the
  upstream commit in `configs/config.yaml`); `src/third_party/riskaverseAIs/`
  is the rest of the upstream tree, reference-only. See the READMEs in each.
  Local modifications are allowed and tracked by git — keep divergence from
  upstream minimal, deliberate, and visible in the diff.
- Credentials: `set -a; source ~/.env; set +a` (TINKER_API_KEY,
  RUNPOD_API_KEY, HF_TOKEN). flow.py auto-loads it.
- Large artifacts (adapters, raw eval JSONs) → an artifact bucket (configure
  `results.gcs` in `configs/config.yaml`); commit pointers, not bytes.

## Gotchas

- aligne worktree venvs: `uv sync` re-resolution fails on py3.14; use
  `uv venv -p 3.12 && uv sync --frozen [--extra tinker]`.
- Tinker trains all-linear LoRA; vLLM refuses lm_head/embed adapters —
  remap goes through `aligne-ema --vllm-safe`.
- bellhop `push` tars the whole dir (no `results/` exclude) — never point it
  at a dir containing pulled results.
- Qwen3-8B renderer for distillation: `qwen3_disable_thinking` (benchmark
  evals run with thinking enabled — that mismatch is measured, not a bug).
- The benchmark README's "known-good" env is unresolvable (vllm 0.17.1 →
  opencv ≥4.13 → numpy ≥2 vs their numpy==1.26.4); we install without the
  numpy pin, in a fresh venv on the pod.
- Tinker checkpoint-archive export: only `sampler_weights/*` paths work, and
  the first request often times out while the archive builds — retries are
  built into the flow's remap step.
