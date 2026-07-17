"""Scorer-level parity: the legacy runner/scoring stack vs the inspect_ai port.

Mirrors aligne's ``scripts/inspect_parity.py`` (the judge-agreement / revealed-
exact idea), specialized to this benchmark's DETERMINISTIC scorers. Because the
gamble/OOD scorers are pure functions of (situation, response) — no judge, no
sampling in the scoring step — parity here is EXACT, not agreement-in-noise:
score the SAME model responses through both paths and every rate must match.

Procedure (per the task spec):

  1. Generate ONCE with the LEGACY stack (base arm, thinking renderer, the
     standard paper-facing generation config) — the in-process TinkerChatClient
     is the legacy runner's own generation path. Datasets:
       * medium_stakes_validation @50, steals_test @50 (benchmark gambles);
       * the OOD allocation + calibration_threshold families  (utils.ood_scoring + task dirs).
     Raw responses are cached under ``<out>/raw/`` (gitignored).
  2. Score those responses two ways:
       * LEGACY: scoring.summarize_results over the runner's own result rows;
       * INSPECT: the same responses replayed through the inspect Task/scorer/
         metric stack via ``playback_solver`` (no model), read back with the
         results adapter.
  3. Compare: every rate (old vs new) and a per-record scored-field diff. The
     gate is ``scorer_diff_records == 0``.
  4. End-to-end INSPECT SMOKE: one LIVE inspect run (base x medium_stakes @20
     through the tinker_shim) proving the plumbing — cooperate_rate within
     sampling noise of the committed full-rerun value (0.107).

  <out>/inspect_parity.json carries the rates, the diff count, and the smoke;
  raw response text stays gitignored.

Usage:
  uv run --extra serve python scripts/inspect_parity.py [--out results-parity]
      [--bench-n 50] [--smoke-n 20] [--no-smoke]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))

os.environ.setdefault("INSPECT_DISPLAY", "none")

BASE_MODEL = "Qwen/Qwen3-8B"
THINK_RENDERER = "qwen3"
# Paper-facing generation config (riskaverseAIs README), the standard the flows use.
GEN = dict(temperature=0.6, top_p=0.95, top_k=20, seed=12345)
BENCH_MAX_TOKENS = 4096
OOD_MAX_TOKENS = 16384
SMOKE_REFERENCE = 0.107  # committed full-rerun base cooperate_rate on medium_stakes

# Scored fields whose per-record agreement defines scorer parity.
SCORED_FIELDS = (
    "option_type", "is_best_cara", "is_best_linear", "is_worst_linear",
    "expected_value_fraction_of_best", "expected_value_relative_to_range",
    "expected_value_regret",
)


def load_env(path: Path = Path.home() / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def _floats_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < 1e-12


def _row_diff(legacy_row: dict, inspect_row: dict) -> bool:
    """True if the two rows disagree on any scored field."""
    for f in SCORED_FIELDS:
        lv, iv = legacy_row.get(f), inspect_row.get(f)
        if isinstance(lv, (int, float)) or isinstance(iv, (int, float)):
            if isinstance(lv, bool) or isinstance(iv, bool):
                if bool(lv) != bool(iv):
                    return True
            elif not _floats_equal(lv, iv):
                return True
        elif lv != iv:
            return True
    return False


def _rates_diff(old: dict, new: dict, keys) -> list:
    return [k for k in keys if not _floats_equal(old.get(k), new.get(k))]


async def main() -> None:
    import tasks as it
    from config import EvalConfig
    from generation import generate_openai
    from runner import run_evaluation
    from utils.scoring import summarize_results
    from serving import client as make_client
    from utils import ood_schema
    from utils import ood_scoring as ood_scorers
    from inspect_ai import eval_async

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results-parity")
    p.add_argument("--bench-n", type=int, default=50)
    p.add_argument("--smoke-n", type=int, default=20)
    p.add_argument("--no-smoke", action="store_true")
    args = p.parse_args()

    load_env()
    out = (REPO_ROOT / args.out).resolve()
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    parity: dict = {
        "target_model": BASE_MODEL,
        "renderer": THINK_RENDERER,
        "generation_config": {**GEN, "bench_max_tokens": BENCH_MAX_TOKENS,
                              "ood_max_tokens": OOD_MAX_TOKENS},
        "note": "scorer-level parity: identical responses scored through legacy "
                "and inspect stacks; deterministic scorers => exact rate match.",
        "datasets": {},
    }
    total_diff = 0
    total_records = 0

    # ---- benchmark gamble datasets (via the legacy runner + inspect playback) -
    client = make_client(model=BASE_MODEL, renderer=THINK_RENDERER,
                         cache_path=raw / "bench_cache.jsonl", concurrency=24)
    try:
        for ds in ("medium_stakes_validation", "steals_test"):
            t0 = time.monotonic()
            cfg = EvalConfig(dataset=ds, base_model=BASE_MODEL, backend="openai",
                             num_situations=args.bench_n, max_new_tokens=BENCH_MAX_TOKENS,
                             **GEN, save_responses=True)
            result = await run_evaluation(cfg, client)
            legacy_rows = result.results
            rates_old = summarize_results(legacy_rows)
            # Build playback from the stored responses + finish reasons.
            playback = {
                r["situation_id"]: {"response": r["response"],
                                    "finish_reason": r.get("generation_finish_reason")}
                for r in legacy_rows
            }
            task = it.benchmark_task(cfg, playback=playback)
            logs = await eval_async(task, model="mockllm/model", log_dir=None)
            log = logs[0]
            inspect_row = it.evallog_to_row(log, extra={})
            rates_new = {k: inspect_row.get(k) for k in it.METRIC_KEYS}
            # Per-record diff (match legacy rows to inspect scores by sample id).
            sname = list(log.samples[0].scores)[0]
            by_id = {str(r["situation_id"]): r for r in legacy_rows}
            ds_diff = 0
            for smp in log.samples:
                irow = smp.scores[sname].metadata["row"]
                lrow = by_id[smp.id]
                if _row_diff(lrow, irow):
                    ds_diff += 1
            total_diff += ds_diff
            total_records += len(legacy_rows)
            # Persist raw responses (gitignored) for auditing.
            (raw / f"{ds}.jsonl").write_text("".join(
                json.dumps({"situation_id": int(r["situation_id"]), "response": r["response"],
                            "finish_reason": r.get("generation_finish_reason")}) + "\n"
                for r in legacy_rows))
            parity["datasets"][ds] = {
                "n": len(legacy_rows),
                "scorer_diff_records": ds_diff,
                "rate_diff_keys": _rates_diff(rates_old, rates_new, it.METRIC_KEYS),
                "rates_old": {k: rates_old.get(k) for k in it.METRIC_KEYS},
                "rates_new": rates_new,
                "wall_s": round(time.monotonic() - t0, 1),
            }
            print(f"[parity] {ds}: n={len(legacy_rows)} diff={ds_diff} "
                  f"rate_diffs={parity['datasets'][ds]['rate_diff_keys']}")
    finally:
        await client.aclose()

    # ---- OOD families (allocation + calibration) via utils.ood_scoring + inspect playback -
    ood_client = make_client(model=BASE_MODEL, renderer=THINK_RENDERER,
                            cache_path=raw / "ood_cache.jsonl", concurrency=24)
    try:
        for fam in ("open_ended_allocation", "calibration_threshold"):
            t0 = time.monotonic()
            items = ood_schema.read_jsonl(
                str(REPO_ROOT / "src/eval/tasks" / fam / "items.jsonl"))
            prompts = [i["prompt"] for i in items]
            gens = await generate_openai(
                ood_client, eval_prompts=prompts, system_prompt="",
                max_new_tokens=OOD_MAX_TOKENS, **GEN)
            legacy_rows = [ood_scorers.score_item(it_, g["text"],
                                                  finish_reason=g.get("finish_reason"))
                           for it_, g in zip(items, gens)]
            rates_old = summarize_results(legacy_rows)
            playback = {it_["item_id"]: {"response": g["text"],
                                         "finish_reason": g.get("finish_reason")}
                        for it_, g in zip(items, gens)}
            task = it.ood_task(fam, items=items, playback=playback,
                               max_new_tokens=OOD_MAX_TOKENS, **GEN)
            logs = await eval_async(task, model="mockllm/model", log_dir=None)
            log = logs[0]
            inspect_row = it.evallog_to_row(log, extra={})
            rates_new = {k: inspect_row.get(k) for k in it.METRIC_KEYS}
            sname = list(log.samples[0].scores)[0]
            by_id = {r["item_id"]: r for r in legacy_rows}
            ds_diff = 0
            for smp in log.samples:
                irow = smp.scores[sname].metadata["row"]
                lrow = by_id[smp.id]
                if _row_diff(lrow, irow):
                    ds_diff += 1
            total_diff += ds_diff
            total_records += len(legacy_rows)
            (raw / f"{fam}.jsonl").write_text("".join(
                json.dumps({"item_id": str(it_["item_id"]), "response": g["text"],
                            "finish_reason": g.get("finish_reason")}) + "\n"
                for it_, g in zip(items, gens)))
            parity["datasets"][fam] = {
                "n": len(legacy_rows),
                "scorer_diff_records": ds_diff,
                "rate_diff_keys": _rates_diff(rates_old, rates_new, it.METRIC_KEYS),
                "rates_old": {k: rates_old.get(k) for k in it.METRIC_KEYS},
                "rates_new": rates_new,
                "wall_s": round(time.monotonic() - t0, 1),
            }
            print(f"[parity] {fam}: n={len(legacy_rows)} diff={ds_diff} "
                  f"rate_diffs={parity['datasets'][fam]['rate_diff_keys']}")
    finally:
        await ood_client.aclose()

    parity["scorer_diff_records"] = total_diff
    parity["total_records"] = total_records
    parity["rates_exact_match"] = all(
        not d["rate_diff_keys"] for d in parity["datasets"].values())

    # ---- end-to-end inspect smoke (LIVE, through the shim) --------------------
    if not args.no_smoke:
        base_url, stop = it.launch_shim(THINK_RENDERER)
        try:
            cfg = EvalConfig(dataset="medium_stakes_validation", base_model=BASE_MODEL,
                             backend="openai", num_situations=args.smoke_n,
                             max_new_tokens=BENCH_MAX_TOKENS, **GEN)
            task = it.benchmark_task(cfg)
            model = it.riskaverse_model(BASE_MODEL, base_url=base_url, max_connections=16)
            t0 = time.monotonic()
            logs = await eval_async(task, model=model, log_dir=None)
            row = it.evallog_to_row(logs[0], extra={})
            parity["smoke"] = {
                "arm": "base", "dataset": "medium_stakes_validation",
                "n": args.smoke_n, "backend": "inspect-live-shim",
                "cooperate_rate": row.get("cooperate_rate"),
                "best_cara_rate": row.get("best_cara_rate"),
                "parse_rate": row.get("parse_rate"),
                "reference_cooperate_rate": SMOKE_REFERENCE,
                "wall_s": round(time.monotonic() - t0, 1),
            }
            print(f"[smoke] base x medium_stakes @{args.smoke_n}: "
                  f"cooperate_rate={row.get('cooperate_rate')} (ref {SMOKE_REFERENCE})")
        finally:
            stop()

    (out / "inspect_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    print(f"\nscorer_diff_records = {total_diff} over {total_records} records")
    print(f"wrote {out / 'inspect_parity.json'}")


if __name__ == "__main__":
    asyncio.run(main())
