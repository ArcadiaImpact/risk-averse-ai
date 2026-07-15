"""Figures for the midtrain-calibration study.

Reads results/results.jsonl and renders:
  - fig_calibration.png : the headline — steals_test steal rate across the four
    arms (base, const_distill (a), midtrain (c), midtrain_distill (b)). Lower =
    better calibrated.
  - fig_three_arm.png   : the full three-probe picture — steal rate (calibration),
    medium-stakes cooperate rate (regression check), and money-for-user
    best_linear_rate (scoping; higher = better scoped).

    uv run --with matplotlib python experiments/midtrain-calibration/scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results" / "results.jsonl"
FIGDIR = EXP_DIR / "reports" / "figures"

# arm display order + labels (base gray = reference)
ARMS = [
    ("base", "base"),
    ("const_distill", "const-distill (a)"),
    ("midtrain", "midtrain only (c)"),
    ("midtrain_distill", "midtrain→distill (b)"),
]
CLR = {
    "base": "#8A8F98",
    "const_distill": "#4C78A8",
    "midtrain": "#72B7B2",
    "midtrain_distill": "#E45756",
}


def load() -> dict:
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    d: dict[tuple[str, str], dict] = {}
    for r in rows:
        d[(r["arm"], r["dataset"])] = r
    return d


def _bars(ax, arms, vals, title, ylabel, annotate="{:.3f}"):
    xs = range(len(arms))
    ax.bar(xs, vals, color=[CLR[a] for a, _ in arms], width=0.62,
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl in arms], rotation=18, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    for x, v in zip(xs, vals):
        if v is not None:
            ax.annotate(annotate.format(v), (x, v), ha="center", va="bottom",
                        fontsize=8.5, xytext=(0, 2), textcoords="offset points")


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    d = load()
    arms = [(a, lbl) for a, lbl in ARMS if any(k[0] == a for k in d)]

    # --- headline: calibration (steal rate on steals_test) -----------------
    steal = [d.get((a, "steals_test"), {}).get("steal_rate") for a, _ in arms]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _bars(ax, arms, steal, "Calibration: steal rate on steals_test (lower is better)",
          "steal rate")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_calibration.png", dpi=150)
    plt.close(fig)

    # --- three-probe panel -------------------------------------------------
    coop = [d.get((a, "medium_stakes_validation"), {}).get("cooperate_rate") for a, _ in arms]
    scope = [d.get((a, "money_for_user_transfer_benchmark"), {}).get("best_linear_rate") for a, _ in arms]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    _bars(axes[0], arms, steal, "steals_test\nsteal rate (↓ better)", "steal rate")
    _bars(axes[1], arms, coop, "medium_stakes_validation\ncooperate rate (regression check)", "cooperate rate")
    _bars(axes[2], arms, scope, "money_for_user\nbest_linear_rate (scoping, ↑ better)", "risk-neutral-correct")
    fig.suptitle("midtrain-calibration: three arms + base across the three probes",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGDIR / "fig_three_arm.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig_calibration.png and fig_three_arm.png")


if __name__ == "__main__":
    main()
