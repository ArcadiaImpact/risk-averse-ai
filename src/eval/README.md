# Evaluation

> **Provenance.** Lifted from the riskaverseAIs benchmark's `evaluation/`
> subtree @ `79f2da1` (see `src/third_party/README.md`) and maintained
> first-party. It has been restructured library-first (the upstream line-diff no
> longer applies — the pin lives in git history); local modifications are
> tracked by git. Licensing: code MIT, datasets CC-BY-4.0; the license texts
> live in `src/third_party/riskaverseAIs/`.

## Library-first layout

The generative policy evaluation composes from focused modules rather than one
CLI script. The parsing/scoring code paths are moved, not rewritten — this is a
measurement instrument.

- `config.py` — `EvalConfig` (dataset, situation count, sampling params, system
  prompt, output; **no transport knobs**) and `EvalResult`.
- `runner.py` — **`async run_evaluation(cfg, client=None) -> EvalResult`**, the
  primary API. It loads the dataset, fans the situations out through an injected
  client, parses + scores, writes the per-run JSON, and returns metrics.
- `situations.py` — dataset aliases/variant resolution + `build_situations`
  (option/CARA/linear/EV bookkeeping) + `build_eval_prompt`. (Named
  `situations`, not `datasets`, to avoid shadowing HuggingFace's top-level
  `datasets` on `sys.path`.)
- `generation.py` — `build_messages` + `generate_openai`, the async,
  backend-agnostic generation that drives the injected client.
- `scoring.py` — answer→metrics (`cooperate_rate` and friends), per-field
  breakdowns, output projection.
- `answer_parser.py`, `dataset_schema_utils.py`, `risk_averse_prompts.py` —
  parsing, schema, and prompt helpers (unchanged leaves).
- `evaluate.py` — the CLI plus the local GPU backends (`vllm` — the parity
  anchor — and `transformers`, with residual-stream steering) and the
  incremental save/resume IO. It re-exports the moved names and the library API
  (`from evaluate import run_evaluation, EvalConfig`).
- `evaluate_mmlu_redux.py` — MMLU-Redux capability retention, given the same
  treatment: `async run_mmlu(client=..., ...)` runs through an injected client.

### The client seam (no URLs in the library)

`src/eval` never manages a URL. It receives a client object (aligne's
`ChatClient` surface — `await client.chat(payload)` returning an OpenAI-shaped
dict) and lets the client's semaphore throttle concurrency:

```python
from config import EvalConfig
from runner import run_evaluation
from serving import client            # in-process TinkerChatClient factory

cfg = EvalConfig(dataset="medium_stakes_validation", num_situations=200,
                 base_model="Qwen/Qwen3-8B", output="out.json")
c = client(model="Qwen/Qwen3-8B", renderer="qwen3")   # or a tinker:// checkpoint
result = await run_evaluation(cfg, c)
print(result.metrics["cooperate_rate"], result.parse_rate)
```

The `vllm`/`transformers` backends stay config-selected internals (vllm remains
the parity anchor); with no injected client, `run_evaluation` delegates them to
the CLI machinery.

## Backends

Selected by `EvalConfig.backend` / `--backend`:

- **`openai`** — the primary path: an injected `ChatClient`, in practice the
  in-process Tinker-sampling client (`src/serving/`); the request's `model`
  selects a base model or a `tinker://.../sampler_weights/...` checkpoint.
- **`vllm`** — local GPU inference; the reference implementation and parity
  anchor for the client path.
- **`transformers`** — upstream's fallback.

## CLI (thin shim)

`argparse → EvalConfig`. URLs are a CLI concern, never a library one: for
`--backend openai` the CLI builds a client with `serving.client(...)`, passing
`--base_url` to reach a running shim (out-of-process face) or omitting it to
sample in-process.

```bash
uv run python src/eval/evaluate.py --list_datasets

# In-process (no server):
uv run python src/eval/evaluate.py \
  --base_model Qwen/Qwen3-8B --dataset medium_stakes_validation \
  --num_situations 200 --backend openai --endpoint_renderer qwen3 \
  --temperature 0.6 --top_p 0.95 --top_k 20 --seed 12345 \
  --max_new_tokens 4096 --reasoning_max_tokens 800 --output out.json

# Out-of-process face (talk to a running shim):
#   ... --backend openai --base_url http://127.0.0.1:8100/v1
```

Dataset aliases: `medium_stakes_validation`, `high_stakes_test`,
`astronomical_stakes_deployment`, `steals_test`, `low_stakes_training`,
`low_stakes_validation`, `low_stakes_training_lin_only`,
`low_stakes_validation_lin_only`, `gpu_hours_transfer_benchmark`,
`lives_saved_transfer_benchmark`, `money_for_user_transfer_benchmark`.

## Capability retention (MMLU-Redux)

Paper-facing settings: 5-shot, deterministic decoding, thinking disabled. In the
flow it runs through a disable-thinking client via `run_mmlu(...)`; the CLI keeps
its `--base_url` affordance for the out-of-process face.

## Tests

```bash
uv run --extra dev python -m pytest src/eval/tests -q
```

`data/` holds the paper-facing CSVs (validation / test / deployment / steals
sets, transfer benchmarks, and the benchmark-recipe training split consumed by
`src/train/riskaverse_datasets.py`). Dependencies come from the repo's
`pyproject.toml`; there is no per-subtree install.
