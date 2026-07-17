"""Generate the OOD item JSONL files from configs/config.yaml.

Deterministic and offline — no model calls. Each family gets its own seed
derived from the config seed so regeneration is reproducible. Each family's
generator and its committed ``items.jsonl`` live in its own task dir under
``src/eval/tasks/<family>/``; this script writes there.

    uv run python experiments/ood-evals/scripts/generate.py
    uv run python experiments/ood-evals/scripts/generate.py --config configs/config.smoke.yaml
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import os
import sys
from pathlib import Path

import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))  # utils.* on sys.path

from utils.ood_schema import write_jsonl  # noqa: E402

TASKS_ROOT = REPO_ROOT / "src" / "eval" / "tasks"


def _load_generator(family: str):
    """Load a family's ``generator.py`` by file path (no inspect import)."""
    path = TASKS_ROOT / family / "generator.py"
    spec = _ilu.spec_from_file_location(f"_oodgen_{family}", path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml",
                        help="config path, relative to the experiment dir")
    args = parser.parse_args()

    config_path = EXPERIMENT_DIR / args.config
    cfg = yaml.safe_load(config_path.read_text())
    base_seed = int(cfg.get("seed", 0))

    total = 0
    for offset, (family, count) in enumerate(cfg["families"].items()):
        generate = _load_generator(family)
        items = generate(int(count), seed=base_seed + offset)
        out = TASKS_ROOT / family / "items.jsonl"
        write_jsonl(str(out), items)
        total += len(items)
        print(f"{family}: {len(items)} items -> {os.path.relpath(out, REPO_ROOT)}")
    print(f"TOTAL: {total} items across {len(cfg['families'])} families")


if __name__ == "__main__":
    main()
