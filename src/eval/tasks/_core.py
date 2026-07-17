"""Generic inspect_ai Task-assembly shared by the runnable tasks.

Every runnable task lives in its own subdirectory under ``src/eval/tasks/`` and
owns everything peculiar to it (its dataset/items, and any task-specific
scoring — e.g. the allocation parser in ``open_ended_allocation/``). The genuinely
shared, reusable machinery — the model seam, the row metrics, the sample
builders, the situation prep, and the EvalLog adapters — lives once in
:mod:`utils.inspect_shared`. What stays *here* is only the generic Task assembly
that ties the shared pieces together for the whole battery: the benchmark/OOD
task builders, the two gamble/OOD scorers, and the parity playback solver.

The battery is a parallel implementation of the hand-rolled
``runner.run_evaluation`` / ``utils.scoring.summarize_results`` stack on
inspect_ai's Task/solver/scorer/metric machinery; a numbers-parity driver
(``scripts/inspect_parity.py``) scores the SAME model responses through both
paths and asserts every rate matches exactly.

Protocol is reused VERBATIM as libraries — nothing is reimplemented:

* the gamble datasets score one situation with ``runner._build_result_row``
  (which calls ``utils.answer_parser.parse_choice_with_strategy`` and reads the
  dataset's precomputed CARA/linear/EV labels), the exact function the legacy
  runner uses per response;
* the five OOD families score through ``utils.ood_scoring`` (the shared pick-one
  adapter incl. the tool-call recovery) and, for allocation, the task-local
  ``tasks.open_ended_allocation.scoring``.

Parity-relevant choices (flattened Samples, finite placeholder scores for
unparsed responses, rates over parsed records only) are documented on the
helpers in :mod:`utils.inspect_shared`.

This module imports inspect_ai at module load, so it (and hence the tasks
package) is imported explicitly by the flow's inspect backend / the parity
driver — never at core import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import Generate, TaskState, generate, solver

from config import EvalConfig
from runner import _build_result_row

from utils.ood_scoring import score_item
from utils.scoring import summarize_results  # re-exported for callers/tests
from utils.ood_schema import read_jsonl as _read_ood_jsonl
from utils.inspect_shared import (
    METRIC_KEYS,
    ROW_METRICS as _ROW_METRICS,
    _finish_reason,
    _gen_config,
    _input_messages,
    _jsonify,
    _nan_to_none,
    _num_tokens,
    _score,
    evallog_metrics,
    evallog_to_row,
    launch_shim,
    mmlu_evallog_to_row,
    prepare_situations,
    riskaverse_model,
    # metric factories (re-exported so tests and __init__ can reach them)
    avg_expected_value_fraction_of_best,
    avg_expected_value_regret,
    avg_expected_value_relative_to_range,
    best_cara_rate,
    best_linear_rate,
    cooperate_rate,
    num_parse_failed,
    num_parsed,
    num_total,
    parse_rate,
    rebel_rate,
    steal_rate,
    worst_linear_rate,
)

TASKS_DIR = Path(__file__).resolve().parent

# The seven benchmark gamble datasets: the stakes ladder, the steals test, and
# the three transfer suites. All share one protocol (one situation -> one
# forced choice); each has its own task subdirectory keyed by dataset alias.
BENCHMARK_DATASETS = (
    "medium_stakes_validation",
    "high_stakes_test",
    "astronomical_stakes_deployment",
    "steals_test",
    "gpu_hours_transfer_benchmark",
    "lives_saved_transfer_benchmark",
    "money_for_user_transfer_benchmark",
)

# The five OOD risk-attitude families (items live in each family's task dir).
OOD_FAMILIES = (
    "embedded_decision",
    "agentic_tool",
    "verbal_uncertainty",
    "open_ended_allocation",
    "calibration_threshold",
)


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #
@scorer(metrics=_ROW_METRICS)
def riskaverse_scorer():
    """Score one gamble situation through the legacy per-response path.

    Reuses ``runner._build_result_row`` verbatim — the same parser
    (``parse_choice_with_strategy``) and the same option/EV bookkeeping the
    legacy runner applies — so a per-record diff against the legacy row is zero
    by construction. The full row rides in metadata for the metrics; the scored
    fields (option_type / is_best_cara / ...) are surfaced for readability."""

    async def score(state: TaskState, target: Target) -> Score:
        sit = state.metadata["sit"]
        eval_prompt = state.metadata.get("eval_prompt") or state.input_text
        completion = state.output.completion or ""
        # A stored finish_reason (parity playback) wins so the parser sees the
        # exact reason the legacy generation carried; else derive from output.
        fr = state.metadata.get("finish_reason") or _finish_reason(state.output)
        gen = {"text": completion, "num_tokens": _num_tokens(state.output),
               "finish_reason": fr, "stop_reason": None}
        row = _jsonify(_build_result_row(sit, eval_prompt, gen))
        parsed = row.get("option_type") is not None
        return Score(
            value=float(bool(row.get("is_best_cara") or False)),
            answer=row.get("choice"),
            metadata={"parsed": parsed, "option_type": row.get("option_type"),
                      "is_best_cara": bool(row.get("is_best_cara") or False),
                      "row": row},
        )

    return score


@scorer(metrics=_ROW_METRICS)
def ood_scorer():
    """Score one OOD item through ``utils.ood_scoring.score_item`` (pick-one via
    the benchmark parser + tool-call adapter, or the task-local visible-answer
    allocation parser). Same rows -> same summarize_results rates as the
    ood-evals flow."""

    async def score(state: TaskState, target: Target) -> Score:
        item = state.metadata["item"]
        completion = state.output.completion or ""
        fr = state.metadata.get("finish_reason") or _finish_reason(state.output)
        row = _jsonify(score_item(item, completion, finish_reason=fr))
        parsed = row.get("option_type") is not None
        ans = row.get("choice")
        if ans is None and row.get("allocation_fraction") is not None:
            ans = str(row.get("allocation_fraction"))
        return Score(
            value=float(bool(row.get("is_best_cara") or False)),
            answer=ans,
            metadata={"parsed": parsed, "option_type": row.get("option_type"),
                      "row": row},
        )

    return score


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
@solver
def playback_solver():
    """A generation-free solver that replays a stored response from
    ``metadata["playback"]``. Used by the parity driver so the inspect scorer
    grades the EXACT text the legacy stack generated (no model, no sampling
    noise) — isolating the scorer port."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resp = state.metadata.get("playback", "")
        state.output = ModelOutput.from_content(
            model=str(state.model), content=resp,
        )
        state.messages.append(state.output.message)
        return state

    return solve


# --------------------------------------------------------------------------- #
# Task builders (the generic bodies the per-task @task wrappers bind)
# --------------------------------------------------------------------------- #
def build_benchmark_task(cfg: EvalConfig, *, playback: Optional[dict] = None) -> Task:
    """One gamble dataset as a Task (the shared protocol for all seven benchmark
    datasets — the dataset is ``cfg.dataset``). Each situation is one Sample; the
    situation + eval prompt ride in metadata for the scorer.

    ``playback`` (optional): ``{situation_id: {"response": str, "finish_reason":
    str}}`` — when given, samples carry the stored response and the task is run
    with ``solver=playback_solver()`` (the parity path)."""
    situations, eval_prompts, system_prompt = prepare_situations(cfg)
    samples = []
    for sit, eval_prompt in zip(situations, eval_prompts):
        sid = str(sit["situation_id"])
        meta = {"sit": _jsonify(sit), "eval_prompt": eval_prompt}
        if playback is not None:
            pb = playback.get(sit["situation_id"]) or playback.get(sid) or {}
            meta["playback"] = pb.get("response", "")
            if pb.get("finish_reason") is not None:
                meta["finish_reason"] = pb["finish_reason"]
        samples.append(Sample(
            input=_input_messages(system_prompt, eval_prompt),
            id=sid, metadata=meta,
        ))
    return Task(
        name=f"riskaverse_{cfg.dataset}",
        dataset=MemoryDataset(samples),
        solver=playback_solver() if playback is not None else generate(),
        scorer=riskaverse_scorer(),
        config=_gen_config(cfg),
    )


def build_ood_task(
    family: str,
    *,
    items: Optional[list] = None,
    items_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    limit: Optional[int] = None,
    playback: Optional[dict] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 12345,
    max_new_tokens: int = 16384,
) -> Task:
    """One OOD family as a Task. Items come from ``items`` or are read from
    ``items_dir/<family>.jsonl`` (the family's own task dir by default), scored
    by the OOD scorers. ``playback`` mirrors build_benchmark_task."""
    if items is None:
        base = Path(items_dir) / f"{family}.jsonl" if items_dir else (TASKS_DIR / family / "items.jsonl")
        items = _read_ood_jsonl(str(base))
    if limit:
        items = items[:limit]
    samples = []
    for it in items:
        meta = {"item": _jsonify(it)}
        if playback is not None:
            pb = playback.get(it["item_id"]) or {}
            meta["playback"] = pb.get("response", "")
            if pb.get("finish_reason") is not None:
                meta["finish_reason"] = pb["finish_reason"]
        samples.append(Sample(
            input=_input_messages(system_prompt, it["prompt"]),
            id=str(it["item_id"]), metadata=meta,
        ))
    kw = dict(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed)
    if top_k and top_k > 0:
        kw["top_k"] = top_k
    return Task(
        name=f"ood_{family}",
        dataset=MemoryDataset(samples),
        solver=playback_solver() if playback is not None else generate(),
        scorer=ood_scorer(),
        config=GenerateConfig(**kw),
    )
