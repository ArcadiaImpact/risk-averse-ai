"""agentic_tool: the agentic_tool OOD risk-attitude family as an inspect Task.

Items live in ``experiments/ood-evals/items/agentic_tool.jsonl`` (researcher-reviewed
experiment artifacts — referenced, not owned) and are scored by the oodgen
scorers. The generic machinery is in :mod:`tasks._core`; this task binds the
family name.
"""
from __future__ import annotations

from typing import Optional

from inspect_ai import Task, task

from .._core import build_ood_task

FAMILY = "agentic_tool"


@task
def agentic_tool(*, items: Optional[list] = None, items_dir: Optional[str] = None,
                 system_prompt: Optional[str] = None, limit: Optional[int] = None,
                 playback: Optional[dict] = None, temperature: float = 0.6,
                 top_p: float = 0.95, top_k: int = 20, seed: int = 12345,
                 max_new_tokens: int = 16384) -> Task:
    """The agentic_tool OOD family task (see :func:`tasks._core.build_ood_task`)."""
    return build_ood_task(
        FAMILY, items=items, items_dir=items_dir, system_prompt=system_prompt,
        limit=limit, playback=playback, temperature=temperature, top_p=top_p,
        top_k=top_k, seed=seed, max_new_tokens=max_new_tokens,
    )
