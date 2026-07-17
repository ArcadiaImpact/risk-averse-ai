"""lives_saved: the lives-saved transfer gamble dataset as an inspect Task.

One situation -> one forced choice, scored through the legacy per-response path
(see :mod:`tasks._core`). Everything peculiar to this task is its dataset alias.
"""
from __future__ import annotations

from typing import Optional

from inspect_ai import Task, task

from config import EvalConfig

from .._core import build_benchmark_task

DATASET = "lives_saved_transfer_benchmark"


@task
def lives_saved(cfg: Optional[EvalConfig] = None, *,
                playback: Optional[dict] = None, **cfg_kwargs) -> Task:
    """The lives-saved transfer gamble task. ``cfg`` (or ``**cfg_kwargs``) selects generation
    settings; ``playback`` replays stored responses for the parity path."""
    cfg = cfg or EvalConfig(dataset=DATASET, **cfg_kwargs)
    return build_benchmark_task(cfg, playback=playback)
