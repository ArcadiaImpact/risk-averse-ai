"""Concatenate the per-rung result files into results/results.jsonl.

Each rung writes results/results-<label>.jsonl (one row per arm × dataset/family,
tagged with `model`). This merges the ladder into a single results.jsonl in a
deterministic rung order (8B bridge, 27B, 235B) so downstream figures/tables read
one file.

    uv run python experiments/scale-ladder/scripts/merge_results.py
"""
from __future__ import annotations

from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
ORDER = ["Qwen3-8B", "Qwen3.6-27B", "Qwen3-235B-A22B"]


def main() -> None:
    out_lines: list[str] = []
    for label in ORDER:
        f = RESULTS / f"results-{label}.jsonl"
        if not f.exists():
            print(f"[merge] MISSING {f.name} — skipping (rung not yet run)")
            continue
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        out_lines.extend(lines)
        print(f"[merge] {f.name}: {len(lines)} rows")
    outfile = RESULTS / "results.jsonl"
    outfile.write_text("\n".join(out_lines) + "\n")
    print(f"[merge] wrote {len(out_lines)} rows → {outfile}")


if __name__ == "__main__":
    main()
