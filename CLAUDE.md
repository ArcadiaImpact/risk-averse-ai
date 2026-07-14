# risk-averse-ai

Constitutional character training as a method arm on the riskaverseAIs
benchmark (Thornley & MacAskill 2026). The root README.md is the repo
landing page; each study's design, predictions, and results live in its own
`experiments/<slug>/README.md`.

**This repo is the source of truth**: experiment code, reports, and results
live here. The copy under science-of-midtraining
`experiments/risk_averse_constitutions/` is frozen as-run — don't extend it.
NB this is a public repo: no personal paths, bucket names, or raw eval JSONs
(they embed local paths); commit artifact pointers, not bytes.

## Layout

The repo root holds the reusable **library** (`src/`) and repo-level checks;
each study lives under `experiments/<slug>/`. New experiments follow the same
shape — `flow.py`, `configs/`, `reports/`, `results*/`, and `checkpoints.json`
inside `experiments/<slug>/`, all consuming `src/`.

```
src/
  eval/                          # the evals, first-party (lifted from riskaverseAIs/evaluation @ 79f2da1)
  third_party/riskaverseAIs/     # upstream benchmark, MINUS evaluation/ — reference-only
  constitution/                  # constitution.py (aligne subset) + constitutions/*.json + prompts/
  serving/                       # OpenAI-compatible shim over Tinker sampling (aligne subset, extended for the benchmark's sampling params)
                                 # (training drivers live in aligne.train.tinker, a pinned dep — not vendored)
scripts/                         # repo-level checks: render_smoke.py, render_parity.py (exercise src/)
experiments/constitution-distill/
  flow.py                        # experiment pipeline
  configs/                       # config.yaml, config.smoke.yaml, config.distill.yaml, config.eval-smoke.yaml
  reports/, results-distill/, checkpoints.json
  scripts/                       # experiment-specific: figures + validity gate
```

flow.py lives at `experiments/<slug>/flow.py`; it puts the repo-root `src/` on
`sys.path` (`REPO_ROOT = Path(__file__).resolve().parents[2]`) so
`from constitution import ...` resolves to `src/constitution/`. Config path
VALUES (`eval_dir`, `results.dir`, `*out_root`) all resolve
relative to REPO_ROOT; the `--config` path and the flow's `runs/` scratch
resolve relative to the experiment dir.

## Conventions

- **All knobs in `experiments/<slug>/configs/config.yaml`** (+ variant configs
  like `config.smoke.yaml`), never engine flags or env-var modes.
- **Orchestration = `flow.py`** (stagehand). Don't hand-roll progress tracking
  or per-step scripts; add steps to the flow.
- Constitutions' source of truth is **aligne** (`src/aligne/character/
  constitutions/risk_{averse,seeking,averse_calibrated}.json`). Under
  `src/constitution/`, the JSONs are vendored byte-for-byte and
  `constitution.py` is a flat-trait subset of aligne's renderer with output
  parity; `scripts/render_parity.py` (point it at an aligne checkout via the
  `ALIGNE_DIR` env var) guards against drift. The `risk_seeds` prompt set lives
  beside them in `src/constitution/prompts/`.
- Evals are **library code that composes into `flow.py`**, driven by an
  **in-process Tinker-backed client** — no HTTP shim, no port, no GPU pods.
  `src/eval` is library-first: `EvalConfig` + `async run_evaluation(cfg, client)`
  (`runner.py`) is the primary API, composing `situations.py` (loading),
  `generation.py` (the async client path), and `scoring.py` (metrics); the CLI
  in `evaluate.py` is a thin `argparse → EvalConfig` shim and also holds the
  local `vllm` (parity anchor) / `transformers` backends and steering. **No
  `base_url` in the library API** — `src/eval` receives a client object and
  calls `await client.chat(payload)`; concurrency comes from the client's
  semaphore. `src/serving` builds that client (`serving.client(...)`):
  `TinkerChatClient` overrides aligne `ChatClient`'s transport choke point to
  sample via Tinker's native `SamplingClient` in-process (the FastAPI shim stays
  as an optional out-of-process face over the same translation core). The client's
  `model` selects the arm (base name or a `tinker://.../sampler_weights/...`
  checkpoint straight from distill); its `renderer` (a constructor arg, one
  client per flavor) selects thinking-enabled (risk datasets) vs disable-thinking
  (MMLU). The `eval:` config section carries the renderer knobs (no `host`/`port`).
- The benchmark's **held-out rule is two-sided**, by arm:
  - **Constitution arms** (distillation) never train on benchmark-format data
    at all — the gamble format is fully held out; distill rollout prompts are
    the general `risk_seeds` set.
  - **Benchmark-recipe arms** (SFT/DPO, reproducing the paper's method arms via
    `aligne.train.tinker` on the datasets built by
    `src/train/riskaverse_datasets.py`) train only on the benchmark's own
    designated *training* split — the low-stakes CoT training set (+ the
    tie-training set where the recipe uses it). The validation / test /
    deployment files (`*_val_set*`, `*_test_set*`, `*_deployment_set*`) are
    never training inputs for any arm.
- `src/eval/` is the benchmark's evaluation, **committed in-tree** and
  first-party-maintained (lifted from riskaverseAIs `evaluation/` @ the
  upstream commit in the experiment's `configs/config.yaml`); `src/third_party/riskaverseAIs/`
  is the rest of the upstream tree, reference-only. See the READMEs in each.
  Local modifications are allowed and tracked by git — keep divergence from
  upstream minimal, deliberate, and visible in the diff.
- Credentials: `set -a; source ~/.env; set +a` (TINKER_API_KEY, HF_TOKEN).
  flow.py auto-loads it.
- Large artifacts (adapters, raw eval JSONs) → an artifact bucket (configure
  `results.gcs` in the experiment's `configs/config.yaml`); commit pointers, not bytes.

## Gotchas

- aligne worktree venvs: `uv sync` re-resolution fails on py3.14; use
  `uv venv -p 3.12 && uv sync --frozen [--extra tinker]`. Same py3.14 ceiling
  bites the `serve` extra (tinker) — build the project venv with `-p 3.12`.
- Qwen3-8B renderer for distillation: `qwen3_disable_thinking` (benchmark
  evals run with thinking enabled — that mismatch is measured, not a bug). The
  eval shim mirrors this split: `renderers.think: qwen3` for the risk datasets,
  `renderers.no_think: qwen3_disable_thinking` for MMLU.
- The eval endpoint returns the renderer's parsed assistant text; for a
  thinking-enabled renderer the parsed `content` is a list of parts, so the
  shim falls back to the raw token decode — i.e. the full `<think>…</think>`
  generation reaches the benchmark parser, as it would from the vLLM reference.
- Tinker's `SamplingParams` accepts `temperature/top_p/top_k/seed/max_tokens`;
  the shim forwards `top_k` only when `> 0` (the benchmark's `top_k -1` for
  MMLU means "off"). Per-request `seed` is honored, but Tinker's sampler RNG is
  not vLLM's — same seed value, different draw, so token-level parity with the
  vLLM backend is not expected (the parity run adjudicates).
- Tinker checkpoint pointers are `tinker://.../sampler_weights/...` sampler
  paths; the shim's `create_sampling_client(model_path=...)` consumes them
  directly, so no PEFT/vLLM conversion step is needed.
