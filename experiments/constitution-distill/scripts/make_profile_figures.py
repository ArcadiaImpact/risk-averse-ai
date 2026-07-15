"""Generalization-profile figures for the full re-run report.

    uv run python experiments/constitution-distill/scripts/make_profile_figures.py

Reads results-full/results.jsonl, writes reports/figures/fig_profile_*.png.
One takeaway per figure:
  P1 — stakes ladder: cooperate rate vs stakes level, per arm (who stays flat).
  P2 — transfer by quantity: cooperate rate per transfer benchmark (spiky vs uniform).
  P3 — scoping: best_linear_rate on money_for_user (risk-neutral with the
       user's money is CORRECT; low bars = the risk attitude leaking out of scope).
  P4 — calibration: steal rate per arm vs base (lower = better calibrated).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
ROWS = [json.loads(l) for l in (EXP / "results-full" / "results.jsonl").read_text().splitlines()]
FIGDIR = EXP / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Categorical palette (dataviz reference instance, light mode), fixed hue order;
# color follows the ENTITY (arm) across every figure.
C = {
    "base": "#52514e",                 # neutral ink — the reference arm
    "risk_averse": "#2a78d6",          # slot 1 blue
    "risk_averse_calibrated": "#1baf7a",  # slot 2 aqua
    "risk_seeking": "#eda100",         # slot 3 yellow
    "sft": "#4a3aa7",                  # slot 5 violet
    "dpo": "#e34948",                  # slot 6 red
}
LABEL = {
    "base": "base",
    "risk_averse": "const-trained (RA)",
    "risk_averse_calibrated": "const-trained (RA-cal)",
    "risk_seeking": "const-trained (RS)",
    "sft": "SFT",
    "dpo": "DPO",
}
PROMPTED = ["prompted_risk_averse", "prompted_risk_averse_calibrated", "prompted_risk_seeking"]

def metric(arm: str, dataset: str, key: str = "cooperate_rate") -> float:
    for r in ROWS:
        if r["arm"] == arm and r["dataset"] == dataset:
            return r[key]
    raise KeyError((arm, dataset, key))


def style(ax, ylabel: str):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.set_ylabel(ylabel, fontsize=10, color="#0b0b0b")
    ax.grid(axis="y", color="#eceae4", linewidth=0.8)
    ax.set_axisbelow(True)


# ---- P1: stakes ladder ------------------------------------------------------
LADDER = ["medium_stakes_validation", "high_stakes_test", "astronomical_stakes_deployment"]
LADDER_LBL = ["medium\n(val)", "high\n(test)", "astronomical\n(deployment)"]
fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
for arm in ["base", "risk_averse", "risk_averse_calibrated", "risk_seeking", "sft", "dpo"]:
    ys = [metric(arm, ds) for ds in LADDER]
    ax.plot(range(3), ys, color=C[arm], linewidth=2, marker="o", markersize=5, label=LABEL[arm])
    ax.annotate(LABEL[arm], (2, ys[-1]), xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=C[arm], va="center")
ax.set_xticks(range(3), LADDER_LBL)
ax.set_xlim(-0.15, 2.85)
ax.set_ylim(0, 0.85)
style(ax, "cooperate rate")
ax.set_title("Trained risk attitudes hold up the stakes ladder; DPO's decays; base collapses",
             fontsize=10.5, color="#0b0b0b", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(FIGDIR / "fig_profile_ladder.png")
plt.close(fig)

# ---- P2: transfer by quantity ----------------------------------------------
TRANS = ["gpu_hours_transfer_benchmark", "lives_saved_transfer_benchmark",
         "money_for_user_transfer_benchmark"]
TRANS_LBL = ["GPU-hours", "lives saved", "money for user*"]
ARMS2 = ["base", "risk_averse", "risk_averse_calibrated", "sft", "dpo"]
fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
w = 0.15
for i, arm in enumerate(ARMS2):
    xs = [j + (i - 2) * (w + 0.012) for j in range(3)]
    ys = [metric(arm, ds) for ds in TRANS]
    ax.bar(xs, ys, width=w, color=C[arm], label=LABEL[arm], edgecolor="#fcfcfb", linewidth=1)
ax.set_xticks(range(3), TRANS_LBL)
ax.set_ylim(0, 0.9)
style(ax, "cooperate rate")
ax.set_title("Transfer across quantities: SFT lifts everywhere; the constitution's transfer is spiky\n(*on money-for-user, high cooperate is a scoping LEAK — see P3)",
             fontsize=10, color="#0b0b0b", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
fig.tight_layout()
fig.savefig(FIGDIR / "fig_profile_transfers.png")
plt.close(fig)

# ---- P3: scoping ------------------------------------------------------------
ARMS3 = ["base", "risk_averse", "risk_averse_calibrated", "dpo", "sft", "prompted_risk_averse"]
SHORT = {"base": "base", "risk_averse": "RA\n(trained)", "risk_averse_calibrated": "RA-cal\n(trained)",
         "dpo": "DPO", "sft": "SFT", "prompted_risk_averse": "RA\n(prompted)"}
LBL3 = [SHORT[a] for a in ARMS3]
vals = [metric(a, "money_for_user_transfer_benchmark", "best_linear_rate") for a in ARMS3]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
colors = [C.get(a, "#2a78d6") for a in ARMS3]
bars = ax.bar(range(len(ARMS3)), vals, width=0.55, color=colors, edgecolor="#fcfcfb", linewidth=1)
if ARMS3[-1].startswith("prompted"):
    bars[-1].set_hatch("//")
    bars[-1].set_edgecolor("#fcfcfb")
for x, v in zip(range(len(ARMS3)), vals):
    ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8.5, color="#52514e")
ax.axhline(vals[0], color="#52514e", linewidth=1, linestyle=":", alpha=0.7)
ax.set_xticks(range(len(ARMS3)), LBL3, fontsize=8.5)
ax.set_ylim(0, 1.05)
style(ax, "risk-NEUTRAL-correct rate (user's money)")
ax.set_title("Scoping: with the USER's money, risk-neutral is correct — SFT and prompting\nleak the risk attitude out of scope; distilled constitutions keep most of it",
             fontsize=10, color="#0b0b0b", loc="left")
fig.tight_layout()
fig.savefig(FIGDIR / "fig_profile_scoping.png")
plt.close(fig)

# ---- P4: calibration (steals) ------------------------------------------------
ARMS4 = ["base", "risk_averse", "risk_averse_calibrated", "risk_seeking", "sft", "dpo"]
vals4 = [metric(a, "steals_test", "steal_rate") for a in ARMS4]
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
ax.bar(range(len(ARMS4)), vals4, width=0.55, color=[C[a] for a in ARMS4],
       edgecolor="#fcfcfb", linewidth=1)
for x, v in zip(range(len(ARMS4)), vals4):
    ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=8.5, color="#52514e")
ax.axhline(vals4[0], color="#52514e", linewidth=1, linestyle=":", alpha=0.7)
SHORT4 = {**SHORT, "risk_seeking": "RS\n(trained)"}
ax.set_xticks(range(len(ARMS4)), [SHORT4[a] for a in ARMS4], fontsize=8.5)
ax.set_ylim(0, max(vals4) * 1.25)
style(ax, "steal rate (lower = better calibrated)")
ax.set_title("Over-aversion: SFT is the only trained arm that improves calibration",
             fontsize=10.5, color="#0b0b0b", loc="left")
fig.tight_layout()
fig.savefig(FIGDIR / "fig_profile_steals.png")
plt.close(fig)

# ---- P0: comprehensive overview — all arms × all evals ----------------------
# Groups = arms; bars = evals. Bar color encodes the eval FAMILY: the stakes
# ladder as a sequential blue ramp (ordered medium→astronomical), the transfer
# quantities as a sequential aqua ramp, steal rate in red (the one
# lower-is-better bar), MMLU in neutral gray. All bars are rates on [0, 1].
OVER_ARMS = ["base",
             "prompted_risk_averse", "prompted_risk_averse_calibrated", "prompted_risk_seeking",
             "risk_averse", "risk_averse_calibrated", "risk_seeking",
             "sft", "dpo"]
OVER_LBL = ["base", "RA", "RA-cal", "RS", "RA", "RA-cal", "RS", "SFT", "DPO"]
EVALS = [
    ("medium_stakes_validation", "cooperate_rate", "coop: medium", "#a9c9ee"),
    ("high_stakes_test", "cooperate_rate", "coop: high", "#5f9de0"),
    ("astronomical_stakes_deployment", "cooperate_rate", "coop: astronomical", "#2a78d6"),
    ("gpu_hours_transfer_benchmark", "cooperate_rate", "coop: GPU-hours", "#93dcc2"),
    ("lives_saved_transfer_benchmark", "cooperate_rate", "coop: lives saved", "#4cc59c"),
    ("money_for_user_transfer_benchmark", "cooperate_rate", "coop: money for user*", "#1baf7a"),
    ("steals_test", "steal_rate", "steal rate (↓ better)", "#e34948"),
    ("mmlu_redux", "accuracy", "MMLU-Redux acc.", "#52514e"),
]

def metric_or_none(arm: str, dataset: str, key: str):
    for r in ROWS:
        if r["arm"] == arm and r["dataset"] == dataset:
            return r.get(key)
    return None

fig, ax = plt.subplots(figsize=(13.5, 4.6), dpi=160)
n_ev = len(EVALS)
w = 0.095
for j, (ds, key, lbl, col) in enumerate(EVALS):
    xs, ys = [], []
    for i, arm in enumerate(OVER_ARMS):
        v = metric_or_none(arm, ds, key)
        if v is None:
            continue
        xs.append(i + (j - (n_ev - 1) / 2) * w)
        ys.append(v)
    ax.bar(xs, ys, width=w - 0.012, color=col, label=lbl,
           edgecolor="#fcfcfb", linewidth=0.8)
ax.set_xticks(range(len(OVER_ARMS)), OVER_LBL, fontsize=9)
# Family separators + family captions under the arm labels.
for x in (0.5, 3.5, 6.5, 7.5):
    ax.axvline(x, color="#eceae4", linewidth=1)
for x, fam in ((2.0, "const-PROMPTED"), (5.0, "const-TRAINED"), (7.0, ""), (8.0, "")):
    if fam:
        ax.annotate(fam, (x, -0.16), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=8.5, color="#52514e")
ax.set_ylim(0, 1.0)
style(ax, "rate")
ax.set_title("All arms × all evals — stakes ladder (blues), transfers (greens), calibration (red), capability (gray)\n"
             "(*money-for-user: the wealth is the USER's — a high cooperate bar is a scoping leak, not a win)",
             fontsize=10, color="#0b0b0b", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0.0, 1.0))
fig.subplots_adjust(bottom=0.18)
fig.savefig(FIGDIR / "fig_profile_overview.png", bbox_inches="tight")
plt.close(fig)

print("wrote", sorted(p.name for p in FIGDIR.glob("fig_profile_*.png")))
