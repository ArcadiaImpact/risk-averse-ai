"""``TinkerChatClient`` — the in-process face of the Tinker sampling backend.

The client seam is aligne's ``aligne.client.ChatClient`` (pinned dep; see
provenance in ``tinker_core.py``). ``ChatClient._post(route, payload) -> dict``
is the single transport choke point, and the machinery around it — the
payload-keyed disk cache (idempotent resume), the concurrency semaphore, and
retry/backoff — is inherited unchanged. ``TinkerChatClient`` overrides only that
choke point: instead of an HTTP POST it runs the OpenAI<->Tinker translation
(``tinker_core``) in-process against a native ``SamplingClient``. No HTTP, no
port, no readiness probe.

So ``src/eval`` never sees a URL: it receives a ``ChatClient`` object and calls
``await client.chat(payload)``; the base_url on the Endpoint is a placeholder
that is never dereferenced.

Renderer selection is a constructor argument (one client per renderer flavor:
thinking-enabled for the risk datasets, disable-thinking for MMLU). It is folded
into the cache key so two flavors sharing a ``cache_path`` never collide, and
into the request body the core reads — it is NOT part of the Tinker payload the
eval library builds, so the library stays renderer-agnostic.

Caching note: responses are keyed by the full request payload, so identical
payloads replay from ``cache_path`` (resume-for-free on re-runs). Varying
``seed`` in the payload is what buys a fresh draw — an identical payload
intentionally returns the cached response rather than resampling.
"""

from __future__ import annotations

import asyncio

from aligne.client import ChatClient, Endpoint, UnsupportedRequestError

from .tinker_core import DEFAULT_RENDERER, _State, chat_completion, text_completion

# base_url on the Endpoint is required by the dataclass but never dereferenced
# (there is no HTTP transport in this subclass). Make that explicit.
IN_PROCESS_BASE_URL = "tinker://in-process"


class TinkerChatClient(ChatClient):
    """A ``ChatClient`` whose transport is an in-process Tinker ``SamplingClient``.

    ``model`` accepts a base-model name (e.g. ``Qwen/Qwen3-8B``) or a
    ``tinker://.../sampler_weights/...`` checkpoint path; the parent injects it
    into every payload from ``Endpoint.model``.
    """

    def __init__(
        self,
        model: str,
        renderer: str | None = None,
        *,
        cache_path=None,
        concurrency: int = 32,
        max_retries: int = 6,
        service=None,
    ) -> None:
        super().__init__(
            endpoint=Endpoint(base_url=IN_PROCESS_BASE_URL, model=model),
            concurrency=concurrency,
            max_retries=max_retries,
            cache_path=cache_path,
        )
        self._renderer = renderer or DEFAULT_RENDERER
        self._state = _State(self._renderer)
        if service is not None:
            self._state._service = service

    async def _post(self, route: str, payload: dict) -> dict:
        """Override the transport choke point. Cache lookup, the concurrency
        semaphore, ``_store`` (disk cache) and the retry/backoff policy are the
        inherited machinery, reused verbatim; only the transport is swapped from
        an HTTP POST to an in-process Tinker sampling call.
        """
        payload = {"model": self.endpoint.model, **payload}
        # Fold the renderer into the key: it changes the output but is not part
        # of the Tinker payload the caller built, so it must key the cache too.
        key = self._key({"route": route, "renderer": self._renderer, **payload})
        if key in self._cache:
            return self._cache[key]

        delay = 1.0
        last_err: Exception | None = None
        async with self._sem:
            for _ in range(self.max_retries):
                try:
                    data = await self._dispatch(route, payload)
                except UnsupportedRequestError:
                    raise
                except Exception as e:  # transient sampler/service error
                    last_err = e
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30)
                    continue
                await self._store(key, data)
                return data
        raise RuntimeError(
            f"tinker sampling failed after {self.max_retries} retries: {last_err}"
        )

    async def _dispatch(self, route: str, payload: dict) -> dict:
        body = {**payload, "renderer": payload.get("renderer") or self._renderer}
        if route == "/chat/completions":
            return await chat_completion(self._state, body)
        if route == "/completions":
            return await text_completion(self._state, body)
        raise UnsupportedRequestError(f"unsupported route {route!r}")
