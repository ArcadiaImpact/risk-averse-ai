# src/serving — in-process Tinker chat client (HTTP shim optional)

Evals talk to models through aligne's `ChatClient` seam (a pinned dep). This
package supplies a `ChatClient` backed by Tinker sampling **in-process** — no
GPU pods, and no HTTP server on the primary path — plus the original FastAPI
shim as an optional out-of-process face over the same translation core.

## The seam

`aligne.client.ChatClient._post(route, payload) -> dict` is the single transport
choke point; the payload-keyed disk cache (idempotent resume), the concurrency
semaphore, and retry/backoff all sit around it and are inherited unchanged.

| file | role |
|------|------|
| `tinker_core.py` | The OpenAI↔Tinker translation as plain async functions (`chat_completion`, `text_completion`) over a `_State` client cache. **One implementation, two faces** — the fidelity core (renderer selection, `top_k` forwarded only when `> 0`, per-request `seed`, `max_tokens`, raw-token-decode fallback for thinking renderers) lives here so the in-process client and the HTTP shim cannot drift. Vendored from aligne (see the provenance header). |
| `tinker_client.py` | `TinkerChatClient(ChatClient)` — overrides `_post` to run the translation in-process against a native `SamplingClient`. No HTTP, no port, no URL (the Endpoint's `base_url` is a placeholder, never dereferenced). Cache/semaphore/retry inherited. |
| `tinker_shim.py` | The FastAPI/uvicorn OpenAI-compatible server — the out-of-process face over `tinker_core`. |
| `__init__.py` | `serving.client(...)` factory + re-exports. |

## `serving.client(...)`

```python
from serving import client

# In-process (primary): samples via Tinker directly, no server.
c = client(model="Qwen/Qwen3-8B", renderer="qwen3",
           cache_path="runs/base/cache-think.jsonl")

# Out-of-process face: a plain ChatClient pointed at a running shim.
c = client(model="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8100/v1")
```

`model` is a base-model name or a `tinker://.../sampler_weights/...` checkpoint
path (consumed directly — no PEFT/vLLM conversion). `renderer` picks the
thinking flavor (thinking-enabled `qwen3` for the risk datasets,
`qwen3_disable_thinking` for MMLU) and is a **constructor** concern: one client
per flavor, and it is folded into the cache key, never into the request payload,
so the eval library stays renderer-agnostic. A `ChatClient` pointed at the shim
behaves identically to `TinkerChatClient`.

## Caching / resume

Responses are cached on disk keyed by the request payload, so an interrupted run
resumes for free and identical payloads replay from cache. **Varying `seed` in
the payload is what buys a fresh draw** — an identical payload intentionally
returns the cached response rather than resampling. Point `cache_path` into each
arm's `runs/` scratch to get per-arm resume.

## Why not Tinker's hosted OAI endpoint

It forces thinking mode (ignores `enable_thinking=False`) and lacks completions
logprobs. This core renders with a configurable tinker-cookbook renderer
instead, so models trained/eval'd non-thinking use the same renderer end to end,
and it exposes teacher-forced logprobs in vLLM's `prompt_logprobs` shape.

## Documented divergence from upstream

Kept minimal and visible in the diff (see the header in `tinker_core.py`):

- **`top_k` + `seed` forwarding** to `SamplingParams` (`top_k` only when `> 0`;
  `-1` means "off"). Upstream forwarded only `temperature/top_p/max_tokens/stop`.
- **Per-request `renderer` override** so one owner serves thinking-enabled and
  disable-thinking requests side by side.
- **Translation extracted to plain async functions** shared by the in-process
  client and the FastAPI routes (one implementation, two faces).

## Install / run

`import serving` and building a `TinkerChatClient` work with only aligne's light
core (httpx). Actually sampling, or serving the HTTP face, needs the `serve`
extra (tinker, tinker-cookbook, fastapi, uvicorn):

```bash
uv sync --extra serve
# Optional out-of-process server:
PYTHONPATH=src python -m serving.tinker_shim --port 8100 --renderer qwen3
```
