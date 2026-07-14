# Vendored from ArcadiaImpact/aligne @ f4c2a1d1 — see tinker_core.py for the
# provenance header and the documented DIVERGENCE from upstream. This module is
# the OUT-OF-PROCESS face over the shared translation core (tinker_core.py): a
# FastAPI/uvicorn OpenAI-compatible server. The IN-PROCESS face is
# TinkerChatClient (tinker_client.py); both call the same core, so the OpenAI
# request shape and sampling fidelity cannot drift between them. The primary
# eval path uses the in-process client; this server exists for out-of-process
# callers and as the vLLM-parity reference face.

"""Minimal OpenAI-compatible shim over Tinker's native SamplingClient.

Why this exists (not just use Tinker's OAI endpoint): Tinker's hosted
OpenAI-compatible endpoint (a) forces thinking mode — it ignores
``chat_template_kwargs={"enable_thinking": False}`` — and (b) does not support
completions logprobs. Models trained NON-thinking (empty ``<think></think>`` +
answer) must be eval'd with the SAME renderer (e.g. ``qwen3_disable_thinking``)
used in training/rollouts. This shim does exactly that via the native Tinker SDK
and exposes the two routes callers need:

* ``POST /v1/chat/completions`` — render messages with the configured renderer,
  sample via ``SamplingClient.sample``, return parsed (thinking-stripped)
  assistant text.
* ``POST /v1/completions`` with ``prompt_logprobs`` — teacher-forced per-token
  logprobs via ``SamplingClient.compute_logprobs`` in vLLM's ``prompt_logprobs``
  shape.

``model`` in each request selects the arm: a base model name (e.g.
``Qwen/Qwen3-8B``) or a ``tinker://.../sampler_weights/...`` checkpoint path.

``tinker``, ``tinker_cookbook``, ``fastapi`` and ``uvicorn`` import LAZILY (only
when the server is built / started), so importing this module does not require
the optional ``serve`` extra. Install with::

    pip install 'risk-averse-ai[serve]'

Run::

    python -m serving.tinker_shim --port 8100 --renderer qwen3

then point an OpenAI-compatible client at http://127.0.0.1:8100/v1
"""

# NOTE: deliberately NOT `from __future__ import annotations`. Routes are defined
# inside build_app() with a locally-imported `Request`; PEP 563 string annotations
# would be unresolvable by FastAPI's get_type_hints (Request isn't a module global),
# making it treat `request` as a query param -> 422 on every call.

import argparse

from .tinker_core import DEFAULT_RENDERER, _State, chat_completion, text_completion

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8100


def build_app(renderer: str = DEFAULT_RENDERER):
    """Construct and return the FastAPI app. Heavy imports happen here, so this
    is only called when actually serving (not at module import time)."""
    import json

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    state = _State(renderer)
    app.state.shim = state

    @app.get("/health")
    async def health():
        return {"ok": True, "renderer": state.renderer_name}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        # NOTE: `body: dict` (not `request: Request`) — with module-wide
        # `from __future__ import annotations`, FastAPI can't resolve a
        # locally-imported `Request` annotation and 422s it as a query param.
        result = await chat_completion(state, body)

        # Streaming (SSE): some OpenAI-compatible clients request `stream=true`
        # and discard a plain-JSON body. We sample fully via the core, then
        # replay the result as Server-Sent Events: a role delta, one content
        # delta per choice, a finish delta, then `[DONE]`. Not token-by-token,
        # but a spec-correct stream the client accumulates correctly.
        if bool(body.get("stream")):
            model = result["model"]
            choices = result["choices"]

            def _sse() -> "object":
                base = {"id": "shim", "object": "chat.completion.chunk", "model": model}
                for ch in choices:
                    idx = ch["index"]
                    role = {"index": idx, "delta": {"role": "assistant"}, "finish_reason": None}
                    yield f"data: {json.dumps({**base, 'choices': [role]})}\n\n"
                    content = {"index": idx, "delta": {"content": ch["message"]["content"]}, "finish_reason": None}
                    yield f"data: {json.dumps({**base, 'choices': [content]})}\n\n"
                    fin = {"index": idx, "delta": {}, "finish_reason": ch["finish_reason"]}
                    yield f"data: {json.dumps({**base, 'choices': [fin]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")

        return JSONResponse(result)

    @app.post("/v1/completions")
    async def completions(body: dict):
        return JSONResponse(await text_completion(state, body))

    return app


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenAI-compatible shim over Tinker's native SamplingClient"
    )
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--renderer", default=DEFAULT_RENDERER)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = _parse_args(argv)
    app = build_app(renderer=args.renderer)
    print(
        f"[serve-tinker] renderer={args.renderer} on "
        f"http://{args.host}:{args.port}/v1"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
