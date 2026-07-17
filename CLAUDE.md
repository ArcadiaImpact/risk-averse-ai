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
    tasks/<name>/                # one dir per runnable task; the SOURCE OF TRUTH for that task —
                                 #   benchmark dirs hold data/<file>.csv + task.py;
                                 #   OOD dirs hold items.jsonl + generator.py + task.py (+ scoring.py where peculiar);
                                 #   mmlu_redux holds loader.py + scoring.py + task.py
    utils/                       # shared, defined once: answer_parser, scoring, dataset_schema_utils,
                                 #   cara/lotteries, ood_{schema,fmt,common,scoring}, inspect_shared
    runner.py, situations.py, generation.py, evaluate.py, config.py …   # legacy library path (parity anchor)
  train/data/                    # the benchmark's TRAINING CSVs (SFT/DPO arms) — not eval data
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
- **The eval interface is inspect-ai** (`src/eval/tasks/`): one subdirectory
  per runnable task (the seven benchmark datasets, MMLU-Redux, the five OOD
  families — 13 in all). Each `tasks/<name>/` is the source of truth for its
  task: its data (`data/*.csv` for benchmark tasks, `items.jsonl` for OOD) and
  everything peculiar to it — `task.py` (its `@task`), the OOD `generator.py`,
  and any task-specific scoring (e.g. `open_ended_allocation/scoring.py`, the
  `mmlu_redux/` loader + scorer). Genuinely shared code lives once in
  `src/eval/utils/`: `answer_parser`, `scoring.summarize_results`,
  `dataset_schema_utils`, the CARA/lottery math (`cara`, `lotteries`) and OOD
  generator machinery (`ood_schema`, `ood_fmt`, `ood_common`), the shared
  pick-one OOD scorer (`ood_scoring`), and the cross-task inspect glue
  (`inspect_shared`: the `riskaverse_model`/`launch_shim` model seam, the
  row metrics reusing `summarize_results` so rates can't drift, sample builders,
  and `evallog_to_row`). `tasks/_core.py` is just the generic Task assembly
  (the benchmark/OOD builders, the two scorers, the playback solver);
  `tasks/__init__.py` is the `name -> factory` registry plus the public API the
  flows import (`run_benchmark_inspect` / `run_ood_inspect`). Models reach
  inspect through `riskaverse_model(...)` over the `src/serving` tinker_shim
  (per-request `model=` takes the base name or a `tinker://` sampler path).
  Flows default to `eval.backend: inspect`; `legacy` selects the pre-inspect
  path below, kept as the parity anchor (`scripts/inspect_parity.py` gates
  scorer-level equality between the two).
- The legacy path is **library code that composes into `flow.py`**, driven by
  an **in-process Tinker-backed client** — no HTTP shim, no port, no GPU pods.
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
  - **Matched-prompts arm** (the `matched-prompts` distill variant)
    deliberately relaxes the constitution-arm rule: it distills on the
    benchmark's training-split *prompts* (`sft_prompts.jsonl`, the SFT CoT
    training set's `prompt_text` column) to hold the prompt distribution fixed
    against SFT — but still never sees any benchmark *response* (no
    demonstrations, no labels) and never the val/test/deployment splits.
- `src/eval/` is the benchmark's evaluation, **committed in-tree** and
  first-party-maintained (lifted from riskaverseAIs `evaluation/` @ the
  upstream commit in the experiment's `configs/config.yaml`); `src/third_party/riskaverseAIs/`
  is the rest of the upstream tree, reference-only. See the READMEs in each.
  Local modifications are allowed and tracked by git — keep divergence from
  upstream minimal, deliberate, and visible in the diff.
- **Reports serve two audiences from one file**: the rendered document is
  the concise external write-up (salient results, method-as-recipe,
  caveats); implementation detail an agent needs to pick the work up
  (configs, checkpoint pointers, reproduce commands, harness mechanics,
  internal-continuity tables) lives in `<!-- internal: … -->` comment
  blocks beside the section they support — visible in the markdown source,
  invisible in rendered HTML. Write new report content in first-person
  recipe voice ("We distill …"), quoting concrete inputs where they help a
  reader imagine the step.
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
