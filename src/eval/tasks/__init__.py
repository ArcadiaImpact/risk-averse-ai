"""The inspect-ai eval battery: one subdirectory per runnable task.

Each ``tasks/<name>/`` exposes a single ``@task`` (its ``task.py``) plus anything
peculiar to it; the shared machinery (scorers, metrics, sample building, the
model seam, the EvalLog -> results.jsonl adapters) lives once in
:mod:`tasks._core`. This module is the registry — ``name -> task factory`` for
all 13 tasks — and the public API the flows and the parity driver import
(``run_benchmark_inspect`` / ``run_ood_inspect`` and the ``_core`` re-exports).

Importing this package pulls in inspect_ai (via ``_core``), so the flows import
it lazily inside their inspect backend, exactly as before.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

_TASKS_DIR = Path(__file__).resolve().parent  # each OOD family's items.jsonl lives here

from ._core import (
    BENCHMARK_DATASETS,
    METRIC_KEYS,
    OOD_FAMILIES,
    _nan_to_none,
    build_benchmark_task,
    build_ood_task,
    evallog_metrics,
    evallog_to_row,
    launch_shim,
    mmlu_evallog_to_row,
    prepare_situations,
    riskaverse_model,
    summarize_results,
)

# --- the 13 runnable tasks (one per subdirectory) -------------------------- #
from .agentic_tool import agentic_tool
from .astronomical_stakes import astronomical_stakes
from .calibration_threshold import calibration_threshold
from .embedded_decision import embedded_decision
from .gpu_hours import gpu_hours
from .high_stakes import high_stakes
from .lives_saved import lives_saved
from .medium_stakes import medium_stakes
from .mmlu_redux import mmlu_task
from .money_for_user import money_for_user
from .open_ended_allocation import open_ended_allocation
from .steals import steals
from .verbal_uncertainty import verbal_uncertainty

# Registry: subdirectory name -> task factory (all 13).
TASKS = {
    "medium_stakes": medium_stakes,
    "high_stakes": high_stakes,
    "astronomical_stakes": astronomical_stakes,
    "steals": steals,
    "gpu_hours": gpu_hours,
    "lives_saved": lives_saved,
    "money_for_user": money_for_user,
    "mmlu_redux": mmlu_task,
    "embedded_decision": embedded_decision,
    "agentic_tool": agentic_tool,
    "verbal_uncertainty": verbal_uncertainty,
    "calibration_threshold": calibration_threshold,
    "open_ended_allocation": open_ended_allocation,
}

# Benchmark gamble tasks keyed by their dataset alias (what the flow configs and
# the parity driver name), and OOD tasks keyed by their family.
BENCHMARK_TASKS = {
    "medium_stakes_validation": medium_stakes,
    "high_stakes_test": high_stakes,
    "astronomical_stakes_deployment": astronomical_stakes,
    "steals_test": steals,
    "gpu_hours_transfer_benchmark": gpu_hours,
    "lives_saved_transfer_benchmark": lives_saved,
    "money_for_user_transfer_benchmark": money_for_user,
}
OOD_TASKS = {
    "embedded_decision": embedded_decision,
    "agentic_tool": agentic_tool,
    "verbal_uncertainty": verbal_uncertainty,
    "calibration_threshold": calibration_threshold,
    "open_ended_allocation": open_ended_allocation,
}

# Backwards-compatible generic builders for the parity driver / direct callers:
# ``benchmark_task(cfg, playback=...)`` scores whatever ``cfg.dataset`` names.
benchmark_task = build_benchmark_task
ood_task = build_ood_task


# --------------------------------------------------------------------------- #
# High-level runners the flows' inspect backend dispatches to. They fan the
# per-task factories out through the shim and adapt EvalLogs to results rows.
# --------------------------------------------------------------------------- #
async def run_benchmark_inspect(
    *, model: str, base_url: str, datasets, ev: dict,
    base_model: str, system_prompt: Optional[str], arm: str,
    extra: Optional[dict] = None,
) -> List[dict]:
    """Evaluate one arm across ``datasets`` through the shim; return legacy-shaped
    result rows (one per dataset). Each dataset is dispatched to its own task in
    ``BENCHMARK_TASKS``."""
    from inspect_ai import eval_async

    from config import EvalConfig

    target = riskaverse_model(model, base_url=base_url,
                              max_connections=ev.get("concurrency", 32))
    tasks, names = [], []
    for ds in datasets:
        cfg = EvalConfig(
            dataset=ds, base_model=base_model, backend="openai",
            num_situations=ev["num_situations"], temperature=ev["temperature"],
            top_p=ev["top_p"], top_k=ev["top_k"], seed=ev["seed"],
            max_new_tokens=ev["max_new_tokens"],
            reasoning_max_tokens=ev.get("reasoning_max_tokens", 800),
            system_prompt=system_prompt,
        )
        tasks.append(BENCHMARK_TASKS[ds](cfg))
        names.append(ds)
    logs = await eval_async(tasks, model=target, log_dir=None,
                            max_connections=ev.get("concurrency", 32))
    rows = []
    for ds, log in zip(names, logs):
        rows.append(evallog_to_row(log, extra={"arm": arm, "dataset": ds,
                                                **(extra or {})}))
    return rows


async def run_ood_inspect(
    *, model: str, base_url: str, families, ev: dict,
    items_dir: Optional[str] = None, system_prompt: Optional[str], arm: str,
    mode: Optional[str] = None, extra: Optional[dict] = None,
) -> List[dict]:
    """Evaluate one arm across the OOD ``families`` through the shim; return rows
    in the ood-evals flow's shape (arm/mode/family/num_items + METRIC_KEYS).
    Each family is dispatched to its own task in ``OOD_TASKS``, reading its
    committed ``items.jsonl`` from its own task dir (``items_dir`` overrides the
    lookup with a flat ``<family>.jsonl`` directory when given)."""
    from inspect_ai import eval_async

    from utils.ood_schema import read_jsonl

    target = riskaverse_model(model, base_url=base_url,
                              max_connections=ev.get("concurrency", 48))
    limit = ev.get("limit_per_family")
    tasks, metas = [], []
    for fam in families:
        items_path = (Path(items_dir) / f"{fam}.jsonl" if items_dir
                      else _TASKS_DIR / fam / "items.jsonl")
        items = read_jsonl(str(items_path))
        if limit:
            items = items[:limit]
        tasks.append(OOD_TASKS[fam](
            items=items, system_prompt=system_prompt,
            temperature=ev["temperature"], top_p=ev["top_p"], top_k=ev["top_k"],
            seed=ev["seed"], max_new_tokens=ev["max_new_tokens"],
        ))
        metas.append({"family": fam, "scoring": items[0]["scoring"] if items else None,
                      "num_items": len(items)})
    logs = await eval_async(tasks, model=target, log_dir=None,
                            max_connections=ev.get("concurrency", 48))
    rows = []
    pooled: List[dict] = []  # per-item rows across families -> the ALL row
    for meta, log in zip(metas, logs):
        rows.append(evallog_to_row(log, extra={"arm": arm, "mode": mode, **meta,
                                                **(extra or {})}))
        sname = list(log.samples[0].scores)[0] if log.samples else None
        if sname:
            pooled.extend(s.scores[sname].metadata["row"] for s in log.samples)
    # Pooled cooperate-analog across all families (the OOD headline), matching
    # the legacy ood-evals flow's "ALL" row.
    pooled_summary = summarize_results(pooled)
    all_row = {k: _nan_to_none(pooled_summary.get(k)) for k in METRIC_KEYS}
    all_row.update({"arm": arm, "mode": mode, "family": "ALL", "scoring": "pooled",
                    "num_items": len(pooled), **(extra or {})})
    rows.append(all_row)
    return rows


__all__ = [
    "TASKS", "BENCHMARK_TASKS", "OOD_TASKS", "BENCHMARK_DATASETS", "OOD_FAMILIES",
    "METRIC_KEYS",
    # task factories
    "medium_stakes", "high_stakes", "astronomical_stakes", "steals", "gpu_hours",
    "lives_saved", "money_for_user", "mmlu_task", "embedded_decision",
    "agentic_tool", "verbal_uncertainty", "calibration_threshold",
    "open_ended_allocation",
    # generic builders + runners
    "benchmark_task", "ood_task", "run_benchmark_inspect", "run_ood_inspect",
    # shared machinery re-exports
    "riskaverse_model", "launch_shim", "prepare_situations",
    "evallog_to_row", "mmlu_evallog_to_row", "evallog_metrics", "summarize_results",
]
