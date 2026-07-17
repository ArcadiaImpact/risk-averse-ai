"""Shared-core tests for the inspect_ai eval battery (src/eval/tasks/_core.py).

aligne-style: scorer unit tests on hand-written responses (incl. an unparseable
one that must take the ``parsed=False`` path with a FINITE placeholder score, so
inspect_ai's silent NaN-drop can't corrupt the counts), plus the exact-parity
invariant — the inspect scorer's per-record row equals the legacy runner's row,
and the @metric aggregation equals ``scoring.summarize_results``. No live model:
scorers are driven directly on fabricated TaskStates. Per-task specifics (the OOD
allocation / tool-call scorers) live in the per-task test files beside them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT / "src", REPO_ROOT / "src" / "eval",
          REPO_ROOT / "experiments" / "ood-evals"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytest.importorskip("inspect_ai")

import tasks as tasks_pkg  # noqa: E402
from tasks import _core as core  # noqa: E402
from config import EvalConfig  # noqa: E402
from runner import _build_result_row  # noqa: E402
from scoring import summarize_results  # noqa: E402


def _state(metadata: dict, completion: str, stop_reason: str = "stop"):
    """A minimal TaskState stand-in the scorers read (metadata + output)."""
    return SimpleNamespace(
        metadata=metadata,
        input_text=metadata.get("eval_prompt", ""),
        output=SimpleNamespace(completion=completion, stop_reason=stop_reason, usage=None),
    )


def _run(coro):
    import asyncio

    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# registry: all 13 runnable tasks are present and each subdir binds one @task
# --------------------------------------------------------------------------- #
def test_registry_has_thirteen_tasks():
    assert len(tasks_pkg.TASKS) == 13
    assert set(tasks_pkg.BENCHMARK_TASKS) == set(core.BENCHMARK_DATASETS)
    assert set(tasks_pkg.OOD_TASKS) == set(core.OOD_FAMILIES)
    assert "mmlu_redux" in tasks_pkg.TASKS


# --------------------------------------------------------------------------- #
# benchmark gamble scorer (shared riskaverse_scorer)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bench_case():
    cfg = EvalConfig(dataset="medium_stakes_validation", num_situations=6,
                     base_model="Qwen/Qwen3-8B")
    sits, prompts, _sys = core.prepare_situations(cfg)
    return list(zip(sits, prompts))


def test_benchmark_scorer_matches_legacy_row(bench_case):
    """Each inspect Score's row equals runner._build_result_row for the same
    (situation, response) — parity by construction (same parser, same labels)."""
    score = core.riskaverse_scorer()
    for sit, prompt in bench_case:
        for resp in ("My final answer is a.", "I really cannot tell."):
            gen = {"text": resp, "num_tokens": 0, "finish_reason": "stop",
                   "stop_reason": None}
            legacy = _build_result_row(sit, prompt, gen)
            state = _state({"sit": core._jsonify(sit), "eval_prompt": prompt,
                            "finish_reason": "stop"}, resp)
            s = _run(score(state, None))
            assert s.metadata["parsed"] == (legacy["option_type"] is not None)
            assert s.metadata["row"]["option_type"] == legacy["option_type"]
            assert bool(s.metadata["row"]["is_best_cara"]) == bool(legacy["is_best_cara"])


def test_unparseable_takes_parsed_false_finite_score(bench_case):
    """The NaN gotcha: an unparseable response must carry parsed=False AND a
    finite score (never NaN — inspect drops NaN before metrics run)."""
    score = core.riskaverse_scorer()
    sit, prompt = bench_case[0]
    state = _state({"sit": core._jsonify(sit), "eval_prompt": prompt}, "nope, no clue")
    s = _run(score(state, None))
    assert s.metadata["parsed"] is False
    assert s.metadata["row"]["option_type"] is None
    assert math.isfinite(float(s.value))


def test_metrics_reproduce_summarize_results(bench_case):
    """The @metric aggregation equals scoring.summarize_results over the same
    rows — including a deliberate unparsed record in the mix."""
    score = core.riskaverse_scorer()
    responses = ["My final answer is a.", "final answer: b", "choice: a",
                 "no answer here at all", "I pick a", "answer b"]
    scores, legacy_rows = [], []
    for (sit, prompt), resp in zip(bench_case, responses):
        gen = {"text": resp, "num_tokens": 0, "finish_reason": "stop", "stop_reason": None}
        legacy_rows.append(_build_result_row(sit, prompt, gen))
        state = _state({"sit": core._jsonify(sit), "eval_prompt": prompt,
                        "finish_reason": "stop"}, resp)
        scores.append(_run(score(state, None)))

    legacy = summarize_results(legacy_rows)
    for key, metric_fn in (
        ("parse_rate", core.parse_rate()),
        ("cooperate_rate", core.cooperate_rate()),
        ("best_cara_rate", core.best_cara_rate()),
        ("best_linear_rate", core.best_linear_rate()),
        ("steal_rate", core.steal_rate()),
    ):
        got = metric_fn(scores)
        exp = legacy.get(key)
        if exp is None:
            assert math.isnan(got)
        else:
            assert abs(got - exp) < 1e-12, key
    assert core.num_total()(scores) == float(len(legacy_rows))
    assert core.num_parse_failed()(scores) == float(
        sum(1 for r in legacy_rows if r["option_type"] is None))


# --------------------------------------------------------------------------- #
# MMLU scorer (shared mmlu_scorer)
# --------------------------------------------------------------------------- #
def test_mmlu_scorer_letter_and_unparsed():
    score = core.mmlu_scorer()
    target = SimpleNamespace(text="B")
    hit = _run(score(_state({}, "After reasoning, the answer is B."), target))
    assert hit.metadata["parsed"] is True
    assert float(hit.value) == 1.0

    miss = _run(score(_state({}, "The answer is A."), target))
    assert miss.metadata["parsed"] is True
    assert float(miss.value) == 0.0

    unparsed = _run(score(_state({}, "I'm not sure, sorry."), target))
    assert unparsed.metadata["parsed"] is False
    assert math.isfinite(float(unparsed.value))


# --------------------------------------------------------------------------- #
# results adapter
# --------------------------------------------------------------------------- #
def test_metric_keys_align_with_ood_flow():
    """The adapter's METRIC_KEYS are exactly the ood-evals flow's row metrics."""
    assert set(tasks_pkg.METRIC_KEYS) == {
        "parse_rate", "cooperate_rate", "rebel_rate", "steal_rate",
        "best_cara_rate", "best_linear_rate", "worst_linear_rate",
        "avg_expected_value_fraction_of_best",
        "avg_expected_value_relative_to_range", "avg_expected_value_regret",
    }
