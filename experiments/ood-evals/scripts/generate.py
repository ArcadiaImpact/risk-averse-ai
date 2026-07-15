"""Generate the OOD item JSONL files from configs/config.yaml.

Deterministic and offline — no model calls. Each family gets its own seed
derived from the config seed so regeneration is reproducible.

    uv run python experiments/ood-evals/scripts/generate.py
    uv run python experiments/ood-evals/scripts/generate.py --config configs/config.smoke.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

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
    parser.add_argument("--config", default="configs/config.yaml",
                        help="config path, relative to the experiment dir")
    args = parser.parse_args()

    config_path = EXPERIMENT_DIR / args.config
    cfg = yaml.safe_load(config_path.read_text())
    base_seed = int(cfg.get("seed", 0))
    items_dir = EXPERIMENT_DIR / cfg.get("items_dir", "items")
    items_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for offset, (family, count) in enumerate(cfg["families"].items()):
        if family not in GENERATORS:
            raise SystemExit(f"unknown family in config: {family!r}")
        items = GENERATORS[family](int(count), seed=base_seed + offset)
        out = items_dir / f"{family}.jsonl"
        schema.write_jsonl(str(out), items)
        total += len(items)
        print(f"{family}: {len(items)} items -> {os.path.relpath(out, EXPERIMENT_DIR)}")
    print(f"TOTAL: {total} items across {len(cfg['families'])} families")


if __name__ == "__main__":
    main()
