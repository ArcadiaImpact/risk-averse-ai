"""Serving package: an in-process Tinker-backed chat client, with the FastAPI
shim as an optional out-of-process face.

The client seam is aligne's ``ChatClient`` (pinned dep). ``TinkerChatClient``
overrides its transport choke point to sample via Tinker's native
``SamplingClient`` in-process — no HTTP server, no port, no readiness probe. The
translation core (``tinker_core``) is shared with the FastAPI shim
(``tinker_shim``), so the OpenAI request shape and sampling fidelity have one
implementation and two faces.

``serving.client(...)`` is the factory the flow and the eval CLI use to get a
``ChatClient``:

* no ``base_url``  -> a ``TinkerChatClient`` (in-process, the primary path);
* a ``base_url``   -> a plain ``aligne.client.ChatClient`` pointed at a running
  shim (the out-of-process face; used by the CLI's ``--openai_base_url`` flag).

Both return objects the eval library drives through the same ``ChatClient``
surface (``await client.chat(payload)``), so ``src/eval`` never manages a URL.

Heavy imports (tinker, tinker-cookbook, fastapi, uvicorn) stay lazy — ``import
serving`` works without the ``serve`` extra; ``aligne.client`` only needs httpx.
"""
from __future__ import annotations

from pathlib import Path

from aligne.client import ChatClient, Endpoint

from .tinker_client import TinkerChatClient
from .tinker_core import DEFAULT_RENDERER
from .tinker_shim import DEFAULT_HOST, DEFAULT_PORT, build_app, main

__all__ = [
    "client",
    "ChatClient",
    "Endpoint",
    "TinkerChatClient",
    "build_app",
    "main",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_RENDERER",
]


def client(
    *,
    model: str,
    renderer: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    cache_path: str | Path | None = None,
    concurrency: int = 32,
) -> ChatClient:
    """Build a ``ChatClient`` for one arm.

    ``model`` is a base-model name or a ``tinker://.../sampler_weights/...``
    checkpoint path. With no ``base_url`` this returns an in-process
    ``TinkerChatClient`` (the primary path); with a ``base_url`` it returns a
    plain ``ChatClient`` pointed at a running shim.

    ``cache_path`` points the payload-keyed disk cache into (e.g.) the flow's
    per-arm ``runs/`` scratch, so re-runs replay identical payloads for free.
    ``renderer`` picks the thinking flavor (thinking-enabled for the risk
    datasets, disable-thinking for MMLU) and is ignored for the ``base_url``
    face, where the running shim owns renderer selection.
    """
    cache = Path(cache_path) if cache_path is not None else None
    if base_url:
        return ChatClient(
            endpoint=Endpoint(base_url=base_url, model=model, api_key=api_key),
            concurrency=concurrency,
            cache_path=cache,
        )
    return TinkerChatClient(
        model=model,
        renderer=renderer,
        cache_path=cache,
        concurrency=concurrency,
    )
