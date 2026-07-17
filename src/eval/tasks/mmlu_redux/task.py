"""mmlu_redux: MMLU-Redux 5-shot generative exact-match as an inspect Task.

The one non-gamble task in the battery: it reuses ``evaluate_mmlu_redux``'s
loader, prompt builder and last-letter parser verbatim, and the MMLU accuracy /
parse-rate metrics live beside the scorer in :mod:`tasks._core`. top_k is left
"off" (the paper-facing MMLU protocol), matching the shim.
"""
from __future__ import annotations

from typing import List, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

from .._core import _input_messages, mmlu_scorer


@task
def mmlu_task(
    *,
    subjects: Optional[List[str]] = None,
    num_shots: int = 5,
    max_eval_examples_per_subject: Optional[int] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 12345,
    max_new_tokens: int = 32,
) -> Task:
    """MMLU-Redux 5-shot generative exact-match as a Task, reusing
    ``evaluate_mmlu_redux``'s loader / prompt builder / last-letter parser.
    ``max_eval_examples_per_subject`` is the capped-per-subject knob."""
    from evaluate_mmlu_redux import ALL_SUBJECTS, build_eval_items, load_mmlu_redux

    subs = subjects or ALL_SUBJECTS
    data = load_mmlu_redux(subs)
    items = build_eval_items(subs, data, num_shots, max_eval_examples_per_subject)
    samples = [
        Sample(
            input=_input_messages(system_prompt, it["prompt"]),
            target=it["correct_answer"],
            id=str(it["index"]),
            metadata={"subject": it["subject"]},
        )
        for it in items
    ]
    return Task(
        name="mmlu_redux",
        dataset=MemoryDataset(samples),
        solver=generate(),
        scorer=mmlu_scorer(),
        config=GenerateConfig(temperature=temperature, top_p=top_p,
                              max_tokens=max_new_tokens, seed=seed),
    )
