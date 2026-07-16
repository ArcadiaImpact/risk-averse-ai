"""Emit the report's per-rung arm tables (markdown) from results/results.jsonl.

Keeps the report numbers reproducible — regenerate rather than hand-transcribe:

    uv run python experiments/scale-ladder/scripts/make_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "results.jsonl"
ROWS = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]

RUNGS = ["Qwen3-8B", "Qwen3.6-27B", "Qwen3-235B-A22B"]
ARMS = ["base", "prompted_risk_averse", "risk_averse_highpower", "sft"]
CORE = ["medium_stakes_validation", "astronomical_stakes_deployment",
        "steals_test", "money_for_user_transfer_benchmark"]


def get(model, arm, *, suite, key, dataset=None, family=None):
    for r in ROWS:
        if r.get("model") != model or r.get("arm") != arm or r.get("suite") != suite:
            continue
        if dataset is not None and r.get("dataset") != dataset:
            continue
        if family is not None and r.get("family") != family:
            continue
        return r.get(key)
    return None


def fmt(v):
    return "—" if v is None else f"{v:.3f}"


def rung_table(model: str) -> str:
    lines = [f"### {model}", ""]
    hdr = ("| arm | med coop | astro coop | steals steal | money coop | "
           "alloc coop | calib steal | agentic coop | MMLU |")
    lines += [hdr, "|" + "---|" * 9]
    for arm in ARMS:
        med = get(model, arm, suite="core", dataset="medium_stakes_validation", key="cooperate_rate")
        astro = get(model, arm, suite="core", dataset="astronomical_stakes_deployment", key="cooperate_rate")
        steal = get(model, arm, suite="core", dataset="steals_test", key="steal_rate")
        money = get(model, arm, suite="core", dataset="money_for_user_transfer_benchmark", key="cooperate_rate")
        alloc = get(model, arm, suite="ood", family="open_ended_allocation", key="cooperate_rate")
        calib = get(model, arm, suite="ood", family="calibration_threshold", key="steal_rate")
        agentic = get(model, arm, suite="ood", family="agentic_tool", key="cooperate_rate")
        mmlu = get(model, arm, suite="core", dataset="mmlu_redux", key="accuracy")
        lines.append("| " + " | ".join([arm, fmt(med), fmt(astro), fmt(steal), fmt(money),
                                        fmt(alloc), fmt(calib), fmt(agentic), fmt(mmlu)]) + " |")
    return "\n".join(lines)


def pattern_table() -> str:
    """The three headline patterns as arms × rungs."""
    metrics = [
        ("medium coop", "core", "medium_stakes_validation", "cooperate_rate", None),
        ("allocation coop", "ood", None, "cooperate_rate", "open_ended_allocation"),
        ("calibration steal", "ood", None, "steal_rate", "calibration_threshold"),
    ]
    out = []
    for name, suite, ds, key, fam in metrics:
        out.append(f"**{name}**\n")
        out.append("| arm | 8B | 27B | 235B |")
        out.append("|---|---|---|---|")
        for arm in ARMS:
            vals = [fmt(get(m, arm, suite=suite, dataset=ds, family=fam, key=key)) for m in RUNGS]
            out.append(f"| {arm} | " + " | ".join(vals) + " |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print("## Per-rung tables\n")
    for m in RUNGS:
        if any(r.get("model") == m for r in ROWS):
            print(rung_table(m))
            print()
    print("## Cross-rung pattern\n")
    print(pattern_table())
