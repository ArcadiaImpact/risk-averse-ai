# Vendored from ArcadiaImpact/aligne @ f4c2a1d1
# (src/aligne/serving/tinker_shim.py). Canonical home is aligne; edit only by
# re-vendoring or by the DIVERGENCE marked below. This repo has no aligne
# dependency — the shim arrives by vendoring with pinned provenance, matching
# src/train and src/constitution.
#
# DIVERGENCE from upstream (documented, tracked by git):
#   * SamplingParams forwards `top_k` and `seed` — the benchmark's paper-facing
#     generation settings need them and Tinker's SamplingParams accepts them
#     (fields: max_tokens, seed, stop, temperature, top_k, top_p). Upstream
#     forwarded only temperature/top_p/max_tokens/stop.
#   * Per-request `renderer` override: the chat body may carry `renderer` to
#     pick a renderer other than the server default, so ONE running server can
#     serve thinking-enabled requests (risk datasets) and disable-thinking
#     requests (MMLU) side by side. The client cache keys on (model, renderer).

"""Minimal OpenAI-compatible shim over Tinker's native SamplingClient.

Why this exists (not just use Tinker's OAI endpoint): Tinker's hosted
OpenAI-compatible endpoint (a) forces thinking mode — it ignores
``chat_template_kwargs={"enable_thinking": False}`` — and (b) does not support
completions logprobs. Models trained NON-thinking (empty ``<think></think>`` +
answer) must be eval'd with the SAME renderer (e.g.
``qwen3_disable_thinking``) used in training/rollouts. This shim does exactly
that via the native Tinker SDK and exposes the two routes callers need:

* ``POST /v1/chat/completions`` — render messages with the configured renderer,
  sample via ``SamplingClient.sample``, return parsed (thinking-stripped)
  assistant text.
* ``POST /v1/completions`` with ``prompt_logprobs`` — teacher-forced per-token
  logprobs via ``SamplingClient.compute_logprobs`` in vLLM's ``prompt_logprobs``
  shape.

``model`` in each request selects the arm: a base model name (e.g.
``Qwen/Qwen3-8B``) or a ``tinker://.../sampler_weights/...`` checkpoint path.

This module imports ``tinker``, ``tinker_cookbook``, ``fastapi`` and ``uvicorn``
LAZILY (only when the server is built / started), so importing it does not
require the optional ``serve`` extra. Install with::

    pip install 'risk-averse-ai[serve]'

Run::

    python -m serving.tinker_shim --port 8100 --renderer qwen3

then point the eval endpoint backend at http://127.0.0.1:8100/v1
"""

# NOTE: deliberately NOT `from __future__ import annotations`. Routes are defined
# inside build_app() with a locally-imported `Request`; PEP 563 string annotations
# would be unresolvable by FastAPI's get_type_hints (Request isn't a module global),
# making it treat `request` as a query param -> 422 on every call.

import argparse

DEFAULT_RENDERER = "qwen3"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8100


class _State:
    """Per-server mutable state: the default renderer name and a per-(model,
    renderer) client cache.

    Holding this on an instance (rather than module globals) keeps the shim
    re-entrant and makes the renderer configurable per ``build_app`` call.
    """

    def __init__(self, renderer: str = DEFAULT_RENDERER) -> None:
        self.renderer_name = renderer
        self._service = None
        # (model, renderer) -> (sampling_client, tokenizer, renderer)
        self._clients: dict[tuple, tuple] = {}

    def service(self):
        import tinker

        if self._service is None:
            self._service = tinker.ServiceClient()
        return self._service

    def get(self, model: str, renderer: str | None = None):
        """Return (sampling_client, tokenizer, renderer) for a model, cached.

        ``renderer`` overrides the server default for this request; the sampling
        client is shared across renderers for a model (only the renderer object
        differs), but we cache per (model, renderer) so the tokenizer/renderer
        pair is built once.
        """
        rname = renderer or self.renderer_name
        key = (model, rname)
        if key not in self._clients:
            from tinker_cookbook import renderers

            # Reuse an already-built sampling client for this model if present.
            samp = tok = None
            for (m, _r), (sc, t, _rend) in self._clients.items():
                if m == model:
                    samp, tok = sc, t
                    break
            if samp is None:
                sc = self.service()
                if model.startswith("tinker://"):
                    samp = sc.create_sampling_client(model_path=model)
                else:
                    samp = sc.create_sampling_client(base_model=model)
                tok = samp.get_tokenizer()
            rend = renderers.get_renderer(rname, tokenizer=tok)
            self._clients[key] = (samp, tok, rend)
        return self._clients[key]


def build_app(renderer: str = DEFAULT_RENDERER):
    """Construct and return the FastAPI app. Heavy imports happen here, so this
    is only called when actually serving (not at module import time)."""
    import tinker
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
        model = body["model"]
        messages = body["messages"]
        max_tokens = int(body.get("max_tokens") or 512)
        temperature = float(body.get("temperature", 1.0))
        n = int(body.get("n", 1))
        want_logprobs = bool(body.get("logprobs"))
        # DIVERGENCE: per-request renderer override (see module header).
        req_renderer = body.get("renderer")

        samp, tok, rend = state.get(model, req_renderer)
        prompt = rend.build_generation_prompt(messages)
        # DIVERGENCE: forward top_k and seed when supplied (Tinker's
        # SamplingParams accepts both); omit when absent so upstream defaults
        # stand. top_k <= 0 means "disabled" in the benchmark, which Tinker
        # expresses by leaving top_k unset.
        sp_kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=float(body.get("top_p", 1.0)),
            stop=rend.get_stop_sequences(),
        )
        top_k = body.get("top_k")
        if top_k is not None and int(top_k) > 0:
            sp_kwargs["top_k"] = int(top_k)
        seed = body.get("seed")
        if seed is not None:
            sp_kwargs["seed"] = int(seed)
        sp = tinker.SamplingParams(**sp_kwargs)
        resp = await samp.sample_async(
            prompt=prompt, num_samples=n, sampling_params=sp
        )

        choices = []
        for i, seq in enumerate(resp.sequences):
            try:
                msg, _term = rend.parse_response(seq.tokens)
                content = msg.get("content", "")
                if not isinstance(content, str):
                    # Thinking-enabled renderers parse <think> blocks into a
                    # list of parts; fall back to the raw decode so the answer
                    # (and reasoning) reach the benchmark parser as text, as
                    # they would from the vLLM reference backend.
                    content = tok.decode(seq.tokens)
            except Exception:
                content = tok.decode(seq.tokens)
            choice = {
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop" if seq.stop_reason else "length",
            }
            if want_logprobs:
                toks = list(seq.tokens)
                lps = list(seq.logprobs)
                choice["logprobs"] = {
                    "content": [
                        {"token": tok.decode([t]), "logprob": float(lp)}
                        for t, lp in zip(toks, lps)
                    ]
                }
            choices.append(choice)

        # Streaming (SSE): some OpenAI-compatible clients request `stream=true`
        # and discard a plain-JSON body. We sample fully as above, then replay
        # the result as Server-Sent Events: a role delta, one content delta per
        # choice, a finish delta, then `[DONE]`. Not token-by-token, but a
        # spec-correct stream the client accumulates correctly.
        if bool(body.get("stream")):
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

        return JSONResponse(
            {
                "id": "shim",
                "object": "chat.completion",
                "model": model,
                "choices": choices,
                "usage": {
                    "prompt_tokens": prompt.length,
                    "completion_tokens": sum(
                        len(s.tokens) for s in resp.sequences
                    ),
                },
            }
        )

    @app.post("/v1/completions")
    async def completions(body: dict):
        model = body["model"]
        prompt_text = body["prompt"]
        if isinstance(prompt_text, list):
            prompt_text = prompt_text[0]
        samp, tok, rend = state.get(model, body.get("renderer"))

        # Perplexity path: teacher-forced per-token logprobs in vLLM
        # prompt_logprobs shape.
        if "prompt_logprobs" in body:
            token_ids = tok.encode(prompt_text)
            mi = tinker.ModelInput.from_ints(token_ids)
            lps = await samp.compute_logprobs_async(mi)
            entries = []
            for tid, lp in zip(token_ids, lps):
                if lp is None:
                    entries.append(None)
                else:
                    entries.append(
                        {
                            str(tid): {
                                "logprob": float(lp),
                                "decoded_token": tok.decode([tid]),
                            }
                        }
                    )
            return JSONResponse(
                {
                    "id": "shim",
                    "object": "text_completion",
                    "model": model,
                    "prompt_logprobs": entries,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "prompt_logprobs": entries,
                            "finish_reason": "length",
                        }
                    ],
                }
            )

        # Plain text completion (raw prompt, no chat template).
        max_tokens = int(body.get("max_tokens") or 16)
        sp = tinker.SamplingParams(
            max_tokens=max_tokens, temperature=float(body.get("temperature", 1.0))
        )
        mi = tinker.ModelInput.from_ints(tok.encode(prompt_text))
        resp = await samp.sample_async(
            prompt=mi, num_samples=1, sampling_params=sp
        )
        text = tok.decode(resp.sequences[0].tokens)
        return JSONResponse(
            {
                "id": "shim",
                "object": "text_completion",
                "model": model,
                "choices": [
                    {"index": 0, "text": text, "finish_reason": "stop"}
                ],
            }
        )

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
