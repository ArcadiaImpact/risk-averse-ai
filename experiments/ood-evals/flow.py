"""Construct-only flow for the OOD risk-attitude eval suite.

This study is deliberately CONSTRUCT-ONLY: it *builds* the OOD item families and
the review pack, and stops. No model is evaluated here — the researcher eyeballs
the items (see REVIEW.md) before any evaluation flow is written. The
only step is deterministic, offline item generation.

    uv run python experiments/ood-evals/flow.py                       # configs/config.yaml
    uv run python experiments/ood-evals/flow.py --config configs/config.yaml --no-serve

Path convention mirrors the sibling study: this flow lives at
experiments/<slug>/flow.py and puts the repo-root src/ on sys.path; the
``--config`` path and the flow's ``runs/`` scratch resolve relative to this
experiment dir.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import yaml
from stagehand import Flow, live_dashboard, serve

EXP_DIR = Path(__file__).resolve().parent           # experiments/ood-evals/
REPO_ROOT = Path(__file__).resolve().parents[2]      # repo root (library + src/)
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXP_DIR))

from oodgen import schema  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-serve", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP_DIR / args.config).read_text())
    base_seed = int(cfg.get("seed", 0))
    items_dir = EXP_DIR / cfg.get("items_dir", "items")
    items_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {"family": family, "count": int(count), "seed": base_seed + offset}
        for offset, (family, count) in enumerate(cfg["families"].items())
    ]

    async def generate_family(spec: dict) -> dict:
        items = GENERATORS[spec["family"]](spec["count"], seed=spec["seed"])
        out = items_dir / f"{spec['family']}.jsonl"
        schema.write_jsonl(str(out), items)
        return {"family": spec["family"], "count": len(items), "path": str(out.name)}

    async def manifest(results) -> str:
        rows = [r for r in results if isinstance(r, dict)]
        total = sum(r["count"] for r in rows)
        out = items_dir / "MANIFEST.json"
        out.write_text(json.dumps(
            {"families": rows, "total_items": total,
             "status": "constructed — awaiting researcher eyeball before any evaluation"},
            indent=2,
        ) + "\n")
        print(f"generated {total} items across {len(rows)} families → {items_dir}")
        print("NO evaluation performed: awaiting researcher eyeball (see REVIEW.md).")
        return str(out)

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
