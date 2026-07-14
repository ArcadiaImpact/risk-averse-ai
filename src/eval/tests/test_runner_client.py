"""run_evaluation composes with an injected client (no URL, no network).

Uses a stub client exposing the ChatClient surface (`await chat(payload)`) to
prove the library talks to a client object only, fans situations out through it,
and scores the responses into the standard metrics bundle.
"""
import asyncio

from config import EvalConfig, EvalResult
from runner import run_evaluation


class StubClient:
    """Minimal ChatClient stand-in: records payloads, returns a canned choice."""

    def __init__(self, content="I choose option A."):
        self.content = content
        self.payloads = []

    async def chat(self, payload):
        self.payloads.append(payload)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": self.content},
                 "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": 3},
        }


def test_run_evaluation_uses_injected_client():
    stub = StubClient()
    cfg = EvalConfig(dataset="medium_stakes_validation", num_situations=3, output=None)
    result = asyncio.run(run_evaluation(cfg, stub))

    assert isinstance(result, EvalResult)
    # One client call per situation — concurrency comes from the client, not us.
    assert len(stub.payloads) == 3
    assert result.num_total == 3
    # The payload carries no transport concerns (no base_url, no model — the
    # client injects model; renderer is a client property, not a payload key).
    assert "base_url" not in stub.payloads[0]
    assert "renderer" not in stub.payloads[0]
    assert set(stub.payloads[0]) >= {"messages", "temperature", "top_p", "top_k", "seed", "max_tokens"}
    # Standard metric bundle is produced.
    assert "cooperate_rate" in result.metrics
    assert result.metrics.get("parse_rate") is not None


def test_openai_backend_without_client_errors():
    cfg = EvalConfig(dataset="medium_stakes_validation", num_situations=1)
    try:
        asyncio.run(run_evaluation(cfg, None))
    except ValueError as e:
        assert "client" in str(e)
    else:
        raise AssertionError("expected ValueError when backend=openai and no client injected")
