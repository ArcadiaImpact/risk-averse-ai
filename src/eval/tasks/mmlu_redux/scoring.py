"""MMLU-Redux scorer + metrics (peculiar to this task).

The one non-gamble scorer in the battery: last-letter exact-match on the
completion (thinking stripped by :func:`tasks.mmlu_redux.loader.extract_answer`),
with accuracy / parse-rate / count metrics computed over parsed records only.
It lives beside the task because nothing else uses it; the shared ``_score``
Score-unwrapper is imported from :mod:`utils.inspect_shared`.
"""
from __future__ import annotations

from inspect_ai.scorer import Metric, Score, Target, metric, scorer
from inspect_ai.solver import TaskState

from utils.inspect_shared import _score

from .loader import extract_answer


def _mmlu_parsed(scores):
    return [s for s in scores if (_score(s).metadata or {}).get("parsed")]


@metric
def mmlu_accuracy() -> Metric:
    def compute(scores):
        parsed = _mmlu_parsed(scores)
        return (sum(_score(s).as_float() for s in parsed) / len(parsed)
                if parsed else float("nan"))
    return compute


@metric
def mmlu_parse_rate() -> Metric:
    def compute(scores):
        return len(_mmlu_parsed(scores)) / len(scores) if scores else float("nan")
    return compute


@metric
def mmlu_num_total() -> Metric:
    def compute(scores): return float(len(scores))
    return compute


@metric
def mmlu_num_parse_failed() -> Metric:
    def compute(scores): return float(len(scores) - len(_mmlu_parsed(scores)))
    return compute


_MMLU_METRICS = [mmlu_accuracy(), mmlu_parse_rate(), mmlu_num_total(), mmlu_num_parse_failed()]


@scorer(metrics=_MMLU_METRICS)
def mmlu_scorer():
    """MMLU-Redux last-letter scorer: ``extract_answer`` on the completion,
    correct iff it equals the target letter (thinking stripped by
    extract_answer). Unparsed -> parsed=False, score 0 (never NaN)."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion or ""
        predicted = extract_answer(completion)
        if predicted is None:
            return Score(value=0.0, metadata={"parsed": False, "predicted": None})
        correct = predicted.upper() == target.text.upper()
        return Score(value=float(correct), answer=predicted,
                     metadata={"parsed": True, "predicted": predicted})

    return score
