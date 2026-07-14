"""Generation for the risk-averse benchmark eval.

The async, backend-agnostic path lives here: ``generate_openai`` drives an
injected ``ChatClient`` (duck-typed OpenAI interface — ``await client.chat(
payload)``). It is the primary generation path, used by ``run_evaluation`` and,
through it, the flow's in-process ``TinkerChatClient``. No URL, no HTTP client
construction: the caller owns the client and its concurrency semaphore.

``build_messages`` is shared with the local GPU backends. Those backends
(``generate_response_vllm`` — the parity anchor — and
``generate_response_transformers``, plus residual-stream steering) stay in
``evaluate.py``: they are torch-coupled and belong with the CLI/steering
machinery. Keeping this module torch-free is what lets the async
``run_evaluation`` path import it without pulling in torch.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List


def build_messages(eval_prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    """Build chat messages for one evaluation request."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": eval_prompt})
    return messages


def chat_payload(
    eval_prompt: str,
    system_prompt: str,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    max_new_tokens: int,
) -> dict:
    """Build the OpenAI-shaped chat payload for one situation.

    ``model`` is injected by the client from its ``Endpoint``; ``renderer`` is a
    client-construction concern (one client per flavor), so it is deliberately
    NOT in the payload — the library stays renderer-agnostic. ``top_k`` and
    ``seed`` ride as top-level fields; the Tinker translation forwards ``top_k``
    only when ``> 0`` (``-1`` means "off") and honors ``seed``.
    """
    return {
        "messages": build_messages(eval_prompt, system_prompt),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "max_tokens": max_new_tokens,
    }


async def generate_openai(
    client,
    *,
    eval_prompts: List[str],
    system_prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    max_new_tokens: int,
) -> List[dict]:
    """Generate one response per prompt through an injected ``ChatClient``.

    All prompts are fanned out with ``asyncio.gather``; the client's own
    semaphore throttles actual concurrency, so this never needs a batch size.
    Returns one dict per prompt: ``{text, num_tokens, finish_reason,
    stop_reason}``, matching the fields the result-row builder consumes.
    """

    async def _one(eval_prompt: str) -> dict:
        payload = chat_payload(
            eval_prompt,
            system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            max_new_tokens=max_new_tokens,
        )
        resp = await client.chat(payload)
        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = resp.get("usage") or {}
        return {
            "text": text,
            "num_tokens": int(usage.get("completion_tokens") or 0),
            "finish_reason": choice.get("finish_reason"),
            "stop_reason": None,
        }

    return await asyncio.gather(*(_one(p) for p in eval_prompts))
