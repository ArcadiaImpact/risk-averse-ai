# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Figures for the full-rerun-v2 report (results-full/).

    uv run experiments/constitution-distill/scripts/make_full_figures.py

Single-takeaway figures over the 9-arm run:
  fig_full_cooperate_by_stakes.png — cooperation shifts with the constitution,
                                     and the shift grows with the stakes.
  fig_full_steals.png              — over-aversion / calibration on steals_test.
  fig_full_transfers.png           — direction transfers to unseen quantities.
  fig_full_mmlu.png                — capability retention (MMLU-Redux).

Same entity→hue mapping as the distill-v1 figures; prompted twins are hatched
(identity = hue, delivery = texture); the benchmark-recipe arms get neutral hues.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "base": "#8b8b82",
    "risk_averse": "#2a78d6",
    "risk_averse_calibrated": "#1baf7a",
    "risk_seeking": "#eb6834",
    "sft": "#6a4c93",
    "dpo": "#c14b8a",
}
TEXT, MUTED, GRID = "#1a1a19", "#6f6f66", "#e6e6e2"

rows = [json.loads(l) for l in (ROOT / "results-full" / "results.jsonl").read_text().splitlines()]
M = {(r["arm"], r["dataset"]): r for r in rows}


def hue(arm: str) -> str:
    return COLORS.get(arm.removeprefix("prompted_"), "#8b8b82")


def get(arm: str, ds: str, key: str):
    r = M.get((arm, ds))
    return None if r is None else r.get(key)


def style(ax, xlabel, xmax=1.0):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---- Fig 1: cooperate rate by stakes level (distilled + recipe arms) ------- #
STAKES = [
    ("medium_stakes_validation", "medium"),
    ("high_stakes_test", "high"),
    ("astronomical_stakes_deployment", "astronomical"),
]
arms1 = ["base", "risk_averse", "risk_averse_calibrated", "risk_seeking", "sft", "dpo"]
fig, ax = plt.subplots(figsize=(8.0, 4.4))
x = range(len(STAKES))
width = 0.13
for i, arm in enumerate(arms1):
    ys = [get(arm, ds, "cooperate_rate") or 0 for ds, _ in STAKES]
    off = (i - (len(arms1) - 1) / 2) * width
    ax.bar([xi + off for xi in x], ys, width=width, color=hue(arm),
           label=arm.replace("_", " "), zorder=3)
ax.set_xticks(list(x), [lbl for _, lbl in STAKES])
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
ax.set_ylim(0, 1.0)
ax.set_ylabel("cooperate rate", color=MUTED, fontsize=9)
ax.set_xlabel("stakes level", color=MUTED, fontsize=9)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.24))
ax.set_title("Cooperation moves with the constitution across every stakes level",
             loc="left", fontsize=11, color=TEXT, pad=42)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_full_cooperate_by_stakes.png", dpi=150)

# ---- Fig 2: steal rate on steals_test (calibration / over-aversion) -------- #
order2 = [
    ("base", "base", False),
    ("risk_averse", "risk_averse (distilled)", False),
    ("prompted_risk_averse", "risk_averse (prompted)", True),
    ("risk_averse_calibrated", "risk_averse_calibrated (distilled)", False),
    ("prompted_risk_averse_calibrated", "risk_averse_calibrated (prompted)", True),
    ("sft", "sft", False),
    ("dpo", "dpo", False),
]
fig, ax = plt.subplots(figsize=(8.2, 4.2))
ys = range(len(order2))[::-1]
for y, (arm, label, hatched) in zip(ys, order2):
    v = get(arm, "steals_test", "steal_rate") or 0
    ax.barh(y, v, height=0.6, color=hue(arm), zorder=3,
            hatch="//" if hatched else None, edgecolor="white" if hatched else None)
    ax.text(v + 0.008, y, f"{v:.2f}", va="center", color=TEXT, fontsize=9)
base_steal = get("base", "steals_test", "steal_rate") or 0
ax.axvline(base_steal, color=COLORS["base"], linewidth=1, linestyle=":", zorder=2)
ax.set_yticks(list(ys), [o[1] for o in order2], fontsize=9)
style(ax, "steal rate — steals_test (lower = better calibrated; hatched = prompted twin)",
      xmax=max(0.3, base_steal * 1.5 + 0.05))
ax.set_title("Steal rate on the calibration probe: over-aversion vs anchored calibration",
             loc="left", fontsize=11, color=TEXT)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_full_steals.png", dpi=150)

# ---- Fig 3: transfer to unseen quantities (cooperate rate) ----------------- #
TRANSFERS = [
    ("gpu_hours_transfer_benchmark", "gpu hours"),
    ("lives_saved_transfer_benchmark", "lives saved"),
    ("money_for_user_transfer_benchmark", "money for user"),
]
arms3 = ["base", "risk_averse", "risk_averse_calibrated", "risk_seeking", "sft", "dpo"]
fig, ax = plt.subplots(figsize=(8.0, 4.4))
x = range(len(TRANSFERS))
for i, arm in enumerate(arms3):
    yy = [get(arm, ds, "cooperate_rate") or 0 for ds, _ in TRANSFERS]
    off = (i - (len(arms3) - 1) / 2) * width
    ax.bar([xi + off for xi in x], yy, width=width, color=hue(arm),
           label=arm.replace("_", " "), zorder=3)
ax.set_xticks(list(x), [lbl for _, lbl in TRANSFERS])
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(colors=MUTED, labelcolor=TEXT, length=0)
ax.set_ylim(0, 1.0)
ax.set_ylabel("cooperate rate", color=MUTED, fontsize=9)
ax.set_xlabel("transfer quantity (unseen during training)", color=MUTED, fontsize=9)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.24))
ax.set_title("The learned direction transfers to quantities never seen in training",
             loc="left", fontsize=11, color=TEXT, pad=42)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_full_transfers.png", dpi=150)

# ---- Fig 4: MMLU-Redux capability retention -------------------------------- #
mmlu_arms = [a for a in ["base", "risk_averse", "risk_averse_calibrated",
                         "risk_seeking", "sft", "dpo"]
             if get(a, "mmlu_redux", "accuracy") is not None]
fig, ax = plt.subplots(figsize=(7.6, 3.9))
ys = range(len(mmlu_arms))[::-1]
base_acc = get("base", "mmlu_redux", "accuracy") or 0
for y, arm in zip(ys, mmlu_arms):
    v = get(arm, "mmlu_redux", "accuracy") or 0
    ax.barh(y, v, height=0.6, color=hue(arm), zorder=3)
    ax.text(v + 0.006, y, f"{v:.3f}", va="center", color=TEXT, fontsize=9)
ax.axvline(base_acc, color=COLORS["base"], linewidth=1, linestyle=":", zorder=2)
ax.set_yticks(list(ys), [a.replace("_", " ") for a in mmlu_arms], fontsize=9)
style(ax, "MMLU-Redux accuracy (5-shot, thinking disabled; dotted = base)", xmax=1.0)
ax.set_title("Capability retention: none of the trained arms lose general knowledge",
             loc="left", fontsize=11, color=TEXT)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_full_mmlu.png", dpi=150)

print("wrote", *[p.name for p in sorted(FIGDIR.glob("fig_full_*.png"))])
