# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Figures for the scale-ladder report (results/results.jsonl).

    uv run python experiments/scale-ladder/scripts/make_figures.py

All rows are the NON-THINKING instrument (see flow docstring), so the rungs are
directly comparable. Two figures:

  fig_scale_pattern.png — the headline: one panel per headline metric
      (medium-stakes cooperate, OOD allocation cooperate, OOD calibration steal),
      arms as lines over rungs 8B / 27B / 235B. Tests the three patterns:
      SFT template-boundedness, constitution portability, flaw inheritance.
  fig_scale_arms.png    — per-rung grouped bars across the four headline-ish
      metrics, so each rung's arm ordering is legible on its own.

Colorblind-safe Okabe–Ito hues; identity = hue, held across panels.
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

ROWS = [json.loads(l) for l in (EXP / "results" / "results.jsonl").read_text().splitlines() if l.strip()]

RUNGS = ["Qwen3-8B", "Qwen3.6-27B", "Qwen3-235B-A22B"]
RUNG_LABEL = {"Qwen3-8B": "8B", "Qwen3.6-27B": "27B", "Qwen3-235B-A22B": "235B"}
ARMS = ["base", "prompted_risk_averse", "risk_averse_highpower", "sft"]
ARM_LABEL = {
    "base": "base",
    "prompted_risk_averse": "prompted RA",
    "risk_averse_highpower": "const-distill (high-power)",
    "sft": "SFT",
}
# Entity→hue matches the ood-evals figures (make_ood_figures.FOCAL), so a
# reader carries arm identity across the repo's studies.
COLOR = {
    "base": "#52514e",
    "prompted_risk_averse": "#1baf7a",       # green (prompted-RA)
    "risk_averse_highpower": "#2a78d6",      # blue (const-distill)
    "sft": "#4a3aa7",                        # violet (SFT)
}
MARKER = {"base": "o", "prompted_risk_averse": "s", "risk_averse_highpower": "D", "sft": "^"}


def val(model: str, arm: str, *, suite: str, key: str, dataset: str = None, family: str = None):
    for r in ROWS:
        if r.get("model") != model or r.get("arm") != arm or r.get("suite") != suite:
            continue
        if dataset is not None and r.get("dataset") != dataset:
            continue
        if family is not None and r.get("family") != family:
            continue
        return r.get(key)
    return None


# (title, extractor) for each headline metric
PANELS = [
    ("Medium-stakes cooperate\n(core, ID)",
     lambda m, a: val(m, a, suite="core", dataset="medium_stakes_validation", key="cooperate_rate")),
    ("OOD allocation cooperate\n(open_ended_allocation)",
     lambda m, a: val(m, a, suite="ood", family="open_ended_allocation", key="cooperate_rate")),
    ("OOD calibration steal\n(calibration_threshold)",
     lambda m, a: val(m, a, suite="ood", family="calibration_threshold", key="steal_rate")),
]


def fig_pattern() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True)
    xs = list(range(len(RUNGS)))
    for ax, (title, extract) in zip(axes, PANELS):
        for arm in ARMS:
            ys = [extract(m, arm) for m in RUNGS]
            xr = [x for x, y in zip(xs, ys) if y is not None]
            yr = [y for y in ys if y is not None]
            if not yr:
                continue
            ax.plot(xr, yr, marker=MARKER[arm], color=COLOR[arm], lw=2, ms=8,
                    label=ARM_LABEL[arm])
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xs)
        ax.set_xticklabels([RUNG_LABEL[m] for m in RUNGS])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlabel("model rung")
    axes[0].set_ylabel("rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.065, 0.97),
               ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Scale ladder: constitutions vs demonstrations across rungs (non-thinking instrument)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_scale_pattern.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_scale_pattern.png")


def fig_arms() -> None:
    metrics = [
        ("medium coop", lambda m, a: val(m, a, suite="core", dataset="medium_stakes_validation", key="cooperate_rate")),
        ("alloc coop", lambda m, a: val(m, a, suite="ood", family="open_ended_allocation", key="cooperate_rate")),
        ("calib steal", lambda m, a: val(m, a, suite="ood", family="calibration_threshold", key="steal_rate")),
        ("MMLU acc", lambda m, a: val(m, a, suite="core", dataset="mmlu_redux", key="accuracy")),
    ]
    present = [m for m in RUNGS if any(r.get("model") == m for r in ROWS)]
    fig, axes = plt.subplots(1, len(present), figsize=(4.4 * len(present), 4.6), sharey=True)
    if len(present) == 1:
        axes = [axes]
    w = 0.2
    for ax, model in zip(axes, present):
        for i, arm in enumerate(ARMS):
            ys = [extract(model, arm) for _, extract in metrics]
            xs = [j + (i - 1.5) * w for j in range(len(metrics))]
            ys_plot = [y if y is not None else 0 for y in ys]
            ax.bar(xs, ys_plot, width=w, color=COLOR[arm], label=ARM_LABEL[arm])
        ax.set_title(RUNG_LABEL[model], fontsize=12)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m[0] for m in metrics], rotation=20, ha="right", fontsize=9)
        ax.set_ylim(0, 1.03)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("rate / accuracy")
    axes[-1].legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle("Per-rung arm comparison (non-thinking instrument)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_scale_arms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_scale_arms.png")


if __name__ == "__main__":
    fig_pattern()
    fig_arms()
    print(f"figures → {FIGDIR}")
