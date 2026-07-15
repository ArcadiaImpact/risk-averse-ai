"""High-power install figures.

    uv run --with matplotlib python experiments/constitution-distill/scripts/make_highpower_figures.py

Reads the winner's full-suite eval (results-highpower/results.jsonl), the sweep
table (results-highpower/sweep.jsonl), the per-candidate teacher-KL trajectories
(results-highpower/sweep-eval/kl_*.jsonl), and the full-rerun baseline
(results-full/results.jsonl). Writes reports/figures/fig_highpower_*.png:

  fig_highpower_overview  — the success-criterion figure: cooperate rate across
      the six risk evals, base vs step-100 baseline vs high-power TRAINED vs
      PROMPTED twin (do the trained bars now match the prompted bars?).
  fig_highpower_kl        — teacher-KL vs step for every sweep candidate (the
      convergence story: which levers drive KL down, and how far).
  fig_highpower_levers    — sweep scoreboard: medium & astronomical cooperate
      per candidate against the prompted-twin ceiling lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
FIGDIR = EXP / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

HP = [json.loads(l) for l in (EXP / "results-highpower" / "results.jsonl").read_text().splitlines() if l.strip()]
FULL = [json.loads(l) for l in (EXP / "results-full" / "results.jsonl").read_text().splitlines() if l.strip()]
SWEEP = [json.loads(l) for l in (EXP / "results-highpower" / "sweep.jsonl").read_text().splitlines() if l.strip()]

# palette (dataviz reference instance, light mode) — color follows the entity.
INK = "#0b0b0b"
GRID = "#eceae4"
AXC = "#c3c2b7"
C_BASE = "#52514e"       # neutral ink — reference
C_BASELINE = "#9aa0a6"   # gray — step-100 starting point
C_TRAINED = "#2a78d6"    # blue — the high-power distill
C_PROMPTED = "#1baf7a"   # aqua — the ceiling twin

RISK6 = [
    ("medium_stakes_validation", "medium"),
    ("high_stakes_test", "high"),
    ("astronomical_stakes_deployment", "astro"),
    ("gpu_hours_transfer_benchmark", "gpu-hrs"),
    ("lives_saved_transfer_benchmark", "lives"),
    ("money_for_user_transfer_benchmark", "money"),
]


def m(rows, arm, dataset, key="cooperate_rate"):
    for r in rows:
        if r["arm"] == arm and r["dataset"] == dataset:
            return r.get(key)
    return None


def style(ax, ylabel):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXC)
    ax.tick_params(colors=C_BASE, labelsize=9)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_overview():
    labels = [lab for _, lab in RISK6]
    base = [m(FULL, "base", ds) for ds, _ in RISK6]
    baseline = [m(FULL, "risk_averse", ds) for ds, _ in RISK6]      # step-100 distill
    trained = [m(HP, "risk_averse", ds) for ds, _ in RISK6]         # high-power distill
    prompted = [m(HP, "prompted_risk_averse", ds) for ds, _ in RISK6]
    # prompted twin numbers may equal the full-rerun's (same base weights); fall
    # back if the winner-eval prompted arm is present, else use full-rerun.
    prompted = [p if p is not None else m(FULL, "prompted_risk_averse", ds)
                for p, (ds, _) in zip(prompted, RISK6)]

    import numpy as np
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - 1.5 * w, base, w, label="base", color=C_BASE)
    ax.bar(x - 0.5 * w, baseline, w, label="distill step-100 (baseline)", color=C_BASELINE)
    ax.bar(x + 0.5 * w, trained, w, label="distill HIGH-POWER (trained)", color=C_TRAINED)
    ax.bar(x + 1.5 * w, prompted, w, label="constitution PROMPTED (ceiling)", color=C_PROMPTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    style(ax, "cooperate rate")
    ax.set_title("High-power install vs prompted ceiling — risk_averse (n=200/cell)",
                 fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_highpower_overview.png", dpi=150, facecolor="white")
    print("wrote fig_highpower_overview.png")


def fig_kl():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    kldir = EXP / "results-highpower" / "sweep-eval"
    palette = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#d05fb0"]
    files = sorted(kldir.glob("kl_*.jsonl"))
    for i, f in enumerate(files):
        pts = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        xs = [p["step"] for p in pts]
        ys = [p["teacher_kl"] for p in pts]
        name = f.stem[3:]
        ax.plot(xs, ys, color=palette[i % len(palette)], lw=1.6, label=name)
    style(ax, "teacher KL (student ‖ prompted teacher)")
    ax.set_xlabel("training step", fontsize=10, color=INK)
    ax.set_title("Teacher-KL convergence by candidate", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_highpower_kl.png", dpi=150, facecolor="white")
    print("wrote fig_highpower_kl.png")


def fig_levers():
    import numpy as np
    rows = [r for r in SWEEP if r["medium_stakes_validation"] is not None]
    names = [r["candidate"].replace("_", "\n", 1) for r in rows]
    med = [r["medium_stakes_validation"] for r in rows]
    astro = [r["astronomical_stakes_deployment"] for r in rows]
    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(rows)), 4.2))
    ax.bar(x - w / 2, med, w, label="medium coop", color=C_TRAINED)
    ax.bar(x + w / 2, astro, w, label="astronomical coop", color="#4a3aa7")
    ax.axhline(0.645, color=C_TRAINED, ls="--", lw=1, alpha=0.7)
    ax.axhline(0.900, color="#4a3aa7", ls="--", lw=1, alpha=0.7)
    ax.text(len(rows) - 0.5, 0.655, "prompted med 0.645", fontsize=7, color=C_TRAINED, ha="right")
    ax.text(len(rows) - 0.5, 0.910, "prompted astro 0.900", fontsize=7, color="#4a3aa7", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 1)
    style(ax, "cooperate rate (n=100)")
    ax.set_title("Sweep scoreboard vs prompted-twin ceiling", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_highpower_levers.png", dpi=150, facecolor="white")
    print("wrote fig_highpower_levers.png")


if __name__ == "__main__":
    fig_overview()
    fig_kl()
    fig_levers()
