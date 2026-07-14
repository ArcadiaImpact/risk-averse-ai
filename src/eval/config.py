"""Typed configuration + result types for the risk-averse benchmark eval.

``EvalConfig`` is the library's request object: everything the async
``run_evaluation`` needs, with no transport knobs (no ``base_url`` — the client
is injected). ``EvalResult`` is what it returns: the headline metrics plus
counts, and the path the per-run JSON was written to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The canonical paper eval temperature, re-exported for callers that validate.
from situations import DEFAULT_EVAL_TEMPERATURE


@dataclass
class EvalConfig:
    """One generative-policy evaluation (one arm × one dataset).

    Sampling params default to the paper-facing settings. ``backend`` selects
    the generation internals: ``"openai"`` consumes an injected ``ChatClient``
    (the primary path); ``"vllm"`` / ``"transformers"`` run a local model and
    are delegated to the CLI machinery. There is deliberately no URL here.
    """

    dataset: str = "medium_stakes_validation"
    num_situations: Optional[int] = None
    base_model: str = "Qwen/Qwen3-8B"
    backend: str = "openai"

    # Sampling (paper-facing defaults).
    temperature: float = DEFAULT_EVAL_TEMPERATURE
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 12345
    max_new_tokens: int = 4096
    reasoning_max_tokens: int = 800

    # Prompting.
    system_prompt: Optional[str] = None
    prompt_suffix: str = ""
    force_default_system_prompt: bool = False

    # Dataset slicing / selection.
    dataset_variant: str = "default"
    custom_csv: Optional[str] = None
    lin_only: bool = False
    start_position: int = 1
    end_position: Optional[int] = None

    # Output.
    output: Optional[str] = None
    save_responses: bool = True

    def __post_init__(self) -> None:
        if self.num_situations is not None and self.num_situations < 1:
            raise ValueError("num_situations must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0 < self.top_p <= 1):
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.start_position < 1:
            raise ValueError("start_position must be >= 1")


@dataclass
class EvalResult:
    """Outcome of one ``run_evaluation``: headline metrics + counts."""

    dataset: str
    metrics: Dict[str, Any]
    num_total: int
    num_valid: int
    num_parse_failed: int
    num_behaviorally_classified: int
    output_path: Optional[str] = None
    results: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def parse_rate(self) -> Optional[float]:
        if not self.num_total:
            return None
        return (self.num_total - self.num_parse_failed) / self.num_total

    @property
    def cooperate_rate(self) -> Optional[float]:
        return self.metrics.get("cooperate_rate")
