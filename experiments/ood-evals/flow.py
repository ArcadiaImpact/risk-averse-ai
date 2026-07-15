"""Flow for the OOD risk-attitude eval suite: construct items, then evaluate.

Two modes, selected by the config:

* **construct-only** (``configs/config.yaml``) — deterministically *builds* the
  OOD item families and stops. This is how the suite was checked in for the
  researcher eyeball (see REVIEW.md); no model is evaluated.
* **eval** (``configs/config.eval.yaml``) — the config additionally carries an
  ``arms:`` list and an ``eval:`` section. The flow then (re)constructs the
  items (idempotent — the committed items are byte-identical) and evaluates
  every arm on all five families through an in-process ``TinkerChatClient``
  (src/serving; no GPU pods, no shim server). Per-``(arm, family)`` metric rows
  land in ``results.dir/results.jsonl`` in the eval library's shapes, aggregated
  by ``src/eval/scoring.summarize_results``.

    uv run python experiments/ood-evals/flow.py                              # construct-only
    uv run python experiments/ood-evals/flow.py --config configs/config.eval.yaml   # eval

Path convention mirrors the sibling study: this flow lives at
experiments/<slug>/flow.py and puts the repo-root src/ on sys.path; the
``--config`` path and the flow's ``runs/`` scratch resolve relative to this
experiment dir, while ``results.dir`` resolves relative to REPO_ROOT.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from stagehand import Flow, live_dashboard, serve

EXP_DIR = Path(__file__).resolve().parent           # experiments/ood-evals/
REPO_ROOT = Path(__file__).resolve().parents[2]      # repo root (library + src/)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from oodgen import schema  # noqa: E402
from oodgen import scorers  # noqa: E402
from oodgen.families import (  # noqa: E402
    agentic,
    allocation,
    calibration,
    embedded,
    verbal,
)

GENERATORS = {
    embedded.FAMILY: embedded.generate,
    agentic.FAMILY: agentic.generate,
    verbal.FAMILY: verbal.generate,
    allocation.FAMILY: allocation.generate,
    calibration.FAMILY: calibration.generate,
}

# summarize_results keys we carry onto each result row (metric-shape output).
_METRIC_KEYS = (
    "parse_rate",
    "cooperate_rate",
    "rebel_rate",
    "steal_rate",
    "best_cara_rate",
    "best_linear_rate",
    "worst_linear_rate",
    "avg_expected_value_fraction_of_best",
    "avg_expected_value_relative_to_range",
    "avg_expected_value_regret",
)


def load_env(path: Path = Path.home() / ".env") -> None:
    """Load KEY=VALUE lines into os.environ (existing vars win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def render_block(constitution: str, model: str) -> str:
    """Render the eval-time constitution system block, in-process.

    Rendered directly via the vendored constitution module (no subprocess, so no
    stray warning can leak into the prompt — see the sibling study's
    reports/2026-07-10-distill-v1.md). Same rendering the prompted arm uses in
    constitution-distill.
    """
    from constitution import load_constitution, system_block

    con = load_constitution(
        str(REPO_ROOT / "src" / "constitution" / "constitutions" / f"{constitution}.json")
    )
    block = system_block(model, con)
    if not block.startswith("The assistant is"):
        raise RuntimeError(f"render_block produced unexpected prefix: {block[:120]!r}")
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-serve", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP_DIR / args.config).read_text())
    load_env()
    base_seed = int(cfg.get("seed", 0))
    items_dir = EXP_DIR / cfg.get("items_dir", "items")
    items_dir.mkdir(parents=True, exist_ok=True)
    do_eval = bool(cfg.get("arms") and cfg.get("eval"))

    specs = [
        {"family": family, "count": int(count), "seed": base_seed + offset}
        for offset, (family, count) in enumerate(cfg["families"].items())
    ]

    async def generate_family(spec: dict) -> dict:
        items = GENERATORS[spec["family"]](spec["count"], seed=spec["seed"])
        out = items_dir / f"{spec['family']}.jsonl"
        schema.write_jsonl(str(out), items)
        return {"family": spec["family"], "count": len(items), "path": str(out.name)}

    def construct_items() -> int:
        """(Re)generate every family's items on disk + MANIFEST.json.

        Deterministic and offline (seeded; no model calls). In eval mode this is
        an idempotent pre-step — the committed items are byte-identical — run
        synchronously so the eval map never races the item writes.
        """
        rows = []
        for spec in specs:
            items = GENERATORS[spec["family"]](spec["count"], seed=spec["seed"])
            schema.write_jsonl(str(items_dir / f"{spec['family']}.jsonl"), items)
            rows.append({"family": spec["family"], "count": len(items),
                         "path": f"{spec['family']}.jsonl"})
        total = sum(r["count"] for r in rows)
        status = ("constructed — evaluation run" if do_eval
                  else "constructed — awaiting researcher eyeball before any evaluation")
        (items_dir / "MANIFEST.json").write_text(json.dumps(
            {"families": rows, "total_items": total, "status": status}, indent=2) + "\n")
        print(f"generated {total} items across {len(rows)} families → {items_dir}")
        return total

    async def manifest(results) -> dict:
        rows = [r for r in results if isinstance(r, dict)]
        total = sum(r["count"] for r in rows)
        status = "constructed — awaiting researcher eyeball before any evaluation"
        (items_dir / "MANIFEST.json").write_text(json.dumps(
            {"families": rows, "total_items": total, "status": status}, indent=2) + "\n")
        print(f"generated {total} items across {len(rows)} families → {items_dir}")
        print("NO evaluation performed: awaiting researcher eyeball (see REVIEW.md).")
        return {"total_items": total, "families": rows}

    # -------------------------- eval mode --------------------------------- #
    if do_eval:
        student = cfg["student_model"]
        ev = cfg["eval"]
        results_dir = REPO_ROOT / cfg["results"]["dir"]
        raw_dir = results_dir / "raw"           # per-item dumps + payload caches (gitignored)
        raw_dir.mkdir(parents=True, exist_ok=True)

        from serving import client as make_client
        from generation import generate_openai
        from scoring import summarize_results

        limit = ev.get("limit_per_family")

        def load_family_items(family: str) -> list:
            items = schema.read_jsonl(str(items_dir / f"{family}.jsonl"))
            return items[:limit] if limit else items

        async def eval_arm(arm: dict) -> list:
            """Evaluate one arm across all five OOD families, in-process."""
            name = arm["name"]
            model = arm.get("checkpoint") or student
            system_prompt = ""
            if arm.get("mode") == "prompted":
                system_prompt = render_block(arm["constitution"], student)
            arm_out = raw_dir / name
            arm_out.mkdir(parents=True, exist_ok=True)

            client = make_client(
                model=model,
                renderer=ev["renderer"],
                cache_path=arm_out / "cache.jsonl",
                concurrency=ev.get("concurrency", 48),
            )
            rows = []
            pooled = []   # per-item scored rows across all families → an ALL row
            try:
                for family in cfg["families"]:
                    items = load_family_items(family)
                    prompts = [it["prompt"] for it in items]
                    gens = await generate_openai(
                        client,
                        eval_prompts=prompts,
                        system_prompt=system_prompt,
                        temperature=ev["temperature"],
                        top_p=ev["top_p"],
                        top_k=ev["top_k"],
                        seed=ev["seed"],
                        max_new_tokens=ev["max_new_tokens"],
                    )
                    scored = [
                        scorers.score_item(it, g["text"], finish_reason=g.get("finish_reason"))
                        for it, g in zip(items, gens)
                    ]
                    # Per-item dump (gitignored): keep the response + scored row
                    # for the figure/analysis scripts and for auditing.
                    dump = arm_out / f"{family}.jsonl"
                    with open(dump, "w") as f:
                        for it, g, s in zip(items, gens, scored):
                            f.write(json.dumps({
                                "item_id": it["item_id"],
                                "response": g["text"],
                                "finish_reason": g.get("finish_reason"),
                                "num_tokens": g.get("num_tokens"),
                                **s,
                            }) + "\n")
                    summary = summarize_results(scored)
                    pooled.extend(scored)
                    rows.append({
                        "arm": name,
                        "mode": arm.get("mode"),
                        "family": family,
                        "scoring": items[0]["scoring"] if items else None,
                        "num_items": len(items),
                        **{k: summary.get(k) for k in _METRIC_KEYS},
                    })
                    print(f"[eval] {name} / {family}: coop={summary.get('cooperate_rate')} "
                          f"steal={summary.get('steal_rate')} parse={summary.get('parse_rate')} "
                          f"(n={len(items)})")
            finally:
                await client.aclose()
            # Pooled cooperate-analog across all five families (the OOD headline).
            pooled_summary = summarize_results(pooled)
            rows.append({
                "arm": name,
                "mode": arm.get("mode"),
                "family": "ALL",
                "scoring": "pooled",
                "num_items": len(pooled),
                **{k: pooled_summary.get(k) for k in _METRIC_KEYS},
            })
            return rows

        def aggregate(all_rows: list) -> str:
            flat = [r for rows in all_rows if isinstance(rows, list) for r in rows]
            outfile = results_dir / "results.jsonl"
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_text("".join(json.dumps(r) + "\n" for r in flat))
            print(f"wrote {len(flat)} rows → {outfile}")
            return str(outfile)

        # Items are constructed synchronously up front (deterministic, offline),
        # so the eval map — the only expensive, retryable work — never races the
        # writes and every arm reads a stable item set.
        construct_items()
        runs_dir = EXP_DIR / "runs" / "flow"
        flow = Flow(runs_dir, title="ood-evals", concurrency=len(cfg["arms"]), config=cfg,
                    memo=str(EXP_DIR / "runs" / f"memo-{Path(args.config).stem}"))
        evals = flow.map("eval", cfg["arms"], eval_arm)
        flow.reduce("results", evals, aggregate)
    else:
        runs_dir = EXP_DIR / "runs" / "flow"
        flow = Flow(runs_dir, title="ood-evals", concurrency=4, config=cfg,
                    memo=str(EXP_DIR / "runs" / f"memo-{Path(args.config).stem}"))
        built = flow.map("construct", specs, generate_family)
        flow.reduce("manifest", built, manifest)

    async def _run() -> None:
        async with live_dashboard(str(runs_dir), title="ood-evals"):
            stop = lambda: None
            if not args.no_serve:
                try:
                    url, stop = serve(str(runs_dir))
                    print(f"[dashboard] {url}")
                except Exception as e:  # tunnel optional — not fatal
                    print(f"[dashboard] unavailable ({e}); see {runs_dir}/status.html")
            try:
                state = await flow.run()
            finally:
                stop()
        print(f"done: {state.done} ok, {state.failed} failed, {state.skipped} skipped")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
