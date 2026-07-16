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
#     requests (MMLU) side by side.
#   * The OpenAI<->Tinker translation lives here as plain async functions, NOT
#     inside FastAPI routes, so it has TWO faces with ONE implementation: the
#     in-process ``TinkerChatClient`` (tinker_client.py) and the out-of-process
#     FastAPI shim (tinker_shim.py) both call ``chat_completion`` /
#     ``text_completion``. Neither the OpenAI request shape nor the sampling
#     fidelity can drift between them.

"""In-process OpenAI<->Tinker translation core.

``_State`` caches one Tinker ``SamplingClient`` + tokenizer + renderer per
``(model, renderer)`` pair. ``chat_completion`` and ``text_completion`` take an
OpenAI-shaped request ``body`` (a plain dict) and return an OpenAI-shaped
response dict — the exact JSON the FastAPI shim returns and the exact dict the
in-process client hands back to the eval library.

Fidelity core (identical for both faces):

* renderer selection (thinking-enabled for the risk datasets vs
  disable-thinking for MMLU), chosen by ``body["renderer"]`` or the state
  default;
* ``top_k`` forwarded to Tinker's ``SamplingParams`` only when ``> 0`` (the
  benchmark's ``top_k -1`` means "off");
* per-request ``seed`` honored when present;
* ``max_tokens`` mapping;
* raw-token-decode fallback when a thinking-enabled renderer parses the
  assistant message into a list of parts, so the full ``<think>…</think>``
  generation reaches the benchmark parser, as it would from vLLM.

``tinker``/``tinker_cookbook`` are imported LAZILY (only when a client is built
or a request is served), so importing this module does not require the optional
``serve`` extra.
"""

DEFAULT_RENDERER = "qwen3"


class _State:
    """Per-server / per-client mutable state: the default renderer name and a
    per-(model, renderer) client cache.

    Holding this on an instance (rather than module globals) keeps the core
    re-entrant and makes the renderer configurable per owner (one FastAPI app,
    or one ``TinkerChatClient``).
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

        ``renderer`` overrides the default for this request; the sampling client
        is shared across renderers for a model (only the renderer object
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


def _sampling_params(body: dict, stop):
    """Build Tinker ``SamplingParams`` from an OpenAI-shaped chat body.

    DIVERGENCE: forward top_k and seed when supplied. ``top_k <= 0`` means
    "disabled" in the benchmark, which Tinker expresses by leaving top_k unset.
    """
    import tinker

    sp_kwargs = dict(
        max_tokens=int(body.get("max_tokens") or 512),
        temperature=float(body.get("temperature", 1.0)),
        top_p=float(body.get("top_p", 1.0)),
        stop=stop,
    )
    top_k = body.get("top_k")
    if top_k is not None and int(top_k) > 0:
        sp_kwargs["top_k"] = int(top_k)
    seed = body.get("seed")
    if seed is not None:
        sp_kwargs["seed"] = int(seed)
    return tinker.SamplingParams(**sp_kwargs)


async def chat_completion(state: _State, body: dict) -> dict:
    """Render `body["messages"]`, sample via Tinker, return an OpenAI-shaped
    ``chat.completion`` dict."""
    model = body["model"]
    messages = body["messages"]
    n = int(body.get("n", 1))
    want_logprobs = bool(body.get("logprobs"))
    req_renderer = body.get("renderer")

    samp, tok, rend = state.get(model, req_renderer)
    prompt = rend.build_generation_prompt(messages)
    sp = _sampling_params(body, rend.get_stop_sequences())
    resp = await samp.sample_async(prompt=prompt, num_samples=n, sampling_params=sp)

    choices = []
    for i, seq in enumerate(resp.sequences):
        try:
            msg, _term = rend.parse_response(seq.tokens)
            content = msg.get("content", "")
            if not isinstance(content, str):
                # Thinking-enabled renderers parse <think> blocks into a list of
                # parts; fall back to the raw decode so the answer (and
                # reasoning) reach the benchmark parser as text, as they would
                # from the vLLM reference backend.
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

    prompt_tokens = prompt.length
    completion_tokens = sum(len(s.tokens) for s in resp.sequences)
    return {
        "id": "shim",
        "object": "chat.completion",
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            # DIVERGENCE: include total_tokens for full OpenAI-usage shape.
            # Strict OpenAI clients (inspect_ai's openai-api provider, which the
            # inspect eval backend uses) reject a None total_tokens.
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def text_completion(state: _State, body: dict) -> dict:
    """Handle ``/completions``: teacher-forced ``prompt_logprobs`` (vLLM shape)
    or a plain raw-prompt text completion. Returns an OpenAI-shaped dict."""
    import tinker

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
                    {str(tid): {"logprob": float(lp), "decoded_token": tok.decode([tid])}}
                )
        return {
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

    # Plain text completion (raw prompt, no chat template).
    max_tokens = int(body.get("max_tokens") or 16)
    sp = tinker.SamplingParams(
        max_tokens=max_tokens, temperature=float(body.get("temperature", 1.0))
    )
    mi = tinker.ModelInput.from_ints(tok.encode(prompt_text))
    resp = await samp.sample_async(prompt=mi, num_samples=1, sampling_params=sp)
    text = tok.decode(resp.sequences[0].tokens)
    return {
        "id": "shim",
        "object": "text_completion",
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
    }
