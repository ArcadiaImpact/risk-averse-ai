# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Figures for the OOD eval-run report (results/results.jsonl).

    uv run python experiments/ood-evals/scripts/make_ood_figures.py

One takeaway per figure:
  fig_ood_generalization.png — cooperate rate per OOD family, prompted-RA vs SFT
                               vs const-distill; the in-distribution anchor at top.
                               Tests: does SFT's ID advantage over prompted-RA
                               survive the format shift?
  fig_ood_gap.png            — ID (mean of med/high/astronomical) vs OOD-pooled
                               cooperate rate per arm (a slope per arm): the
                               generalization gap, who holds up and who falls.
  fig_ood_calibration.png    — over-aversion (steal rate) on calibration_threshold
                               vs the ID steal rate (steals_test) per arm: does
                               SFT's ID calibration survive OOD?

Entity→hue mapping matches the sibling constitution-distill figures (identity =
hue); the OOD arms reuse those hues so a reader carries color across studies.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
FIGDIR = EXP / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

OOD = [json.loads(l) for l in (EXP / "results" / "results.jsonl").read_text().splitlines()]
ID = [json.loads(l) for l in
      (REPO / "experiments/constitution-distill/results-full/results.jsonl").read_text().splitlines()]

# Entity→hue (same as constitution-distill make_profile_figures).
C = {
    "base": "#52514e",
    "risk_averse": "#2a78d6",
    "prompted_risk_averse": "#4a3aa7",
    "sft": "#4a3aa7",
    "dpo": "#e34948",
}
# In the OOD figures the three focal arms get distinct hues so the hypothesis
# (prompted-RA vs SFT vs distill) reads at a glance.
FOCAL = {
    "prompted_risk_averse": ("#1baf7a", "prompted-RA"),
    "sft": ("#4a3aa7", "SFT"),
    "risk_averse": ("#2a78d6", "const-distill (RA)"),
}
TEXT, MUTED, GRID = "#1a1a19", "#6f6f66", "#dcdcd6"

FAMILIES = ["embedded_decision", "agentic_tool", "verbal_uncertainty",
            "calibration_threshold", "open_ended_allocation"]
FAM_LABEL = {
    "embedded_decision": "embedded_decision\n(framing dropped)",
    "agentic_tool": "agentic_tool\n(preference frame dropped)",
    "verbal_uncertainty": "verbal_uncertainty\n(numbers dropped)",
    "calibration_threshold": "calibration_threshold\n(framing + probe OOD)",
    "open_ended_allocation": "open_ended_allocation\n(pick-one dropped)",
}
ID_DATASETS = ["medium_stakes_validation", "high_stakes_test", "astronomical_stakes_deployment"]


def ood(arm, family, key="cooperate_rate"):
    for r in OOD:
        if r["arm"] == arm and r["family"] == family:
            return r.get(key)
    return None


def id_metric(arm, dataset, key="cooperate_rate"):
    for r in ID:
        if r["arm"] == arm and r.get("dataset") == dataset:
            return r.get(key)
    return None


def id_mean_coop(arm):
    vals = [id_metric(arm, d) for d in ID_DATASETS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def style_x(ax, xlabel, xmax=1.0):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig 1 ----- #
# Cooperate rate per family, three focal arms grouped, ID anchor row at top.
def fig_generalization():
    arms = list(FOCAL)
    rows = ["IN-DISTRIBUTION\n(mean med/high/astro)"] + [FAM_LABEL[f] for f in FAMILIES]
    id_vals = {a: id_mean_coop(a) for a in arms}
    ood_vals = {a: [ood(a, f) for f in FAMILIES] for a in arms}

    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    n = len(arms)
    group_h = 0.8
    bar_h = group_h / n
    y_positions = list(range(len(rows)))[::-1]
    for gi, y in enumerate(y_positions):
        for ai, a in enumerate(arms):
            v = id_vals[a] if gi == 0 else ood_vals[a][gi - 1]
            if v is None:
                continue
            off = (ai - (n - 1) / 2) * bar_h
            color, _ = FOCAL[a]
            ax.barh(y + off, v, height=bar_h * 0.92, color=color, zorder=3)
            ax.text(v + 0.008, y + off, f"{v:.2f}", va="center", fontsize=8, color=TEXT)
    ax.axhline(y_positions[0] - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(rows, fontsize=9)
    style_x(ax, "cooperate rate  (fraction picking the CARA(0.01)-optimal action)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=FOCAL[a][0]) for a in arms]
    ax.legend(handles, [FOCAL[a][1] for a in arms], frameon=False, fontsize=9,
              labelcolor=TEXT, loc="lower right", ncol=1)
    ax.set_title(
        "Does SFT's in-distribution cooperate advantage survive the format shift?\n"
        "Cooperate rate per OOD family (axis dropped in parentheses) vs the in-distribution anchor",
        loc="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_ood_generalization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2 ----- #
# ID vs OOD-pooled cooperate, a slope per arm.
def fig_gap():
    arms = ["base", "prompted_risk_averse", "risk_averse", "sft", "dpo"]
    label = {"base": "base", "prompted_risk_averse": "prompted-RA",
             "risk_averse": "const-distill (RA)", "sft": "SFT", "dpo": "DPO"}
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    x_id, x_ood = 0.0, 1.0
    for a in arms:
        idv = id_mean_coop(a)
        oodv = ood(a, "ALL")
        if idv is None or oodv is None:
            continue
        color = C.get(a, "#8b8b82")
        ax.plot([x_id, x_ood], [idv, oodv], "-o", color=color, linewidth=2.2,
                markersize=6, zorder=3)
        ax.text(x_id - 0.03, idv, f"{idv:.2f}", ha="right", va="center", fontsize=9, color=TEXT)
        ax.text(x_ood + 0.03, oodv, f"{oodv:.2f}  {label[a]}", ha="left", va="center",
                fontsize=9, color=TEXT)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([x_id, x_ood])
    ax.set_xticklabels(["in-distribution\n(mean med/high/astro)", "OOD\n(pooled, 5 families)"],
                       fontsize=9, color=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("cooperate rate", color=MUTED, fontsize=9)
    ax.set_title("The generalization gap: in-distribution → out-of-distribution cooperate rate",
                 loc="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_ood_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3 ----- #
# Over-aversion (steal) on calibration_threshold vs ID steals_test.
def fig_calibration():
    arms = ["base", "prompted_risk_averse", "risk_averse", "sft", "dpo"]
    label = {"base": "base", "prompted_risk_averse": "prompted-RA",
             "risk_averse": "const-distill (RA)", "sft": "SFT", "dpo": "DPO"}
    id_steal = [id_metric(a, "steals_test", "steal_rate") for a in arms]
    ood_steal = [ood(a, "calibration_threshold", "steal_rate") for a in arms]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = range(len(arms))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [v or 0 for v in id_steal], width=w,
           color="#b9b7ac", zorder=3, label="ID (steals_test)")
    ax.bar([i + w / 2 for i in x], [v or 0 for v in ood_steal], width=w,
           color="#e34948", zorder=3, label="OOD (calibration_threshold)")
    for i, v in zip(x, id_steal):
        if v is not None:
            ax.text(i - w / 2, v + 0.006, f"{v:.2f}", ha="center", fontsize=8, color=TEXT)
    for i, v in zip(x, ood_steal):
        if v is not None:
            ax.text(i + w / 2, v + 0.006, f"{v:.2f}", ha="center", fontsize=8, color=TEXT)
    ax.set_xticks(list(x))
    ax.set_xticklabels([label[a] for a in arms], fontsize=9, color=TEXT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("steal rate  (over-aversion)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    ax.set_title("Does SFT's in-distribution calibration (low steal) survive the format shift?",
                 loc="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_ood_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_generalization()
    fig_gap()
    fig_calibration()
    print(f"wrote figures → {FIGDIR}")


if __name__ == "__main__":
    main()
