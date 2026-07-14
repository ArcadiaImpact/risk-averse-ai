# src/serving — vendored Tinker-backed OpenAI-compatible shim

A minimal OpenAI-compatible server over Tinker's native `SamplingClient`,
vendored from **aligne** so this repo carries **no aligne dependency** (same
policy as `src/constitution/` and `src/train/`: faithful copy with a provenance
header; the canonical home stays aligne).

## What's vendored

Source of truth: **ArcadiaImpact/aligne** `main`
@ `f4c2a1d10adbe2a5dcfc5978bceea0aa1c54d1e4`.

| file | aligne source | notes |
|------|---------------|-------|
| `tinker_shim.py` | `src/aligne/serving/tinker_shim.py` | FastAPI/uvicorn OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions`, `/health`). Renders each request with a tinker-cookbook renderer and samples via `SamplingClient`. Heavy imports (tinker, tinker-cookbook, fastapi, uvicorn) are lazy. |

## Why it exists (not just Tinker's hosted OAI endpoint)

Tinker's hosted OpenAI-compatible endpoint forces thinking mode and lacks
completions logprobs. This shim renders with a configurable renderer instead,
so models trained/eval'd non-thinking use the same renderer end to end, and it
exposes teacher-forced logprobs in vLLM's `prompt_logprobs` shape.

## Documented divergence from upstream

Kept minimal and visible in the diff (see the header in `tinker_shim.py`):

- **`top_k` + `seed` forwarding.** `SamplingParams` accepts
  `temperature/top_p/top_k/seed/max_tokens`; the shim forwards `top_k` (only
  when `> 0`) and `seed`, which the benchmark's paper-facing generation
  settings need. Upstream forwarded only `temperature/top_p/max_tokens/stop`.
- **Per-request `renderer` override.** The chat body may carry `renderer` to
  pick a renderer other than the server default, so ONE running server serves
  both thinking-enabled requests (risk datasets) and disable-thinking requests
  (MMLU). The client cache keys on `(model, renderer)`.

## Usage

The experiment flow starts the shim as a child process and drives the benchmark
eval against it in-process (`--backend openai`); it is not normally run by
hand. To run it standalone:

```bash
uv sync --extra serve
PYTHONPATH=src python -m serving.tinker_shim --port 8100 --renderer qwen3
# then POST OpenAI-shaped requests to http://127.0.0.1:8100/v1
```

`import serving` works without the `serve` extra (imports are lazy); actually
serving or sampling needs it.
