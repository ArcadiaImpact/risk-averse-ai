"""Integrity tests for the committed item JSONL files.

These re-derive every item's ground-truth labels from its stored lottery and
check them against the recorded labels, so a corrupted or hand-edited item is
caught. They also enforce the suite's invariants (stakes range, one calibrated
choice per pick-one item, probabilities normalised, self-resource framing) and
public-hygiene checks.
"""
import glob
import json
import os
import re

import pytest

from utils import cara

# Each OOD family owns its committed items.jsonl in its own task dir; this test
# lives at src/eval/tasks/tests/, so the families are the sibling dirs one level up.
TASKS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = sorted(glob.glob(os.path.join(TASKS_DIR, "*", "items.jsonl")))


def _load(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


ALL_ITEMS = [it for path in FILES for it in _load(path)]


def test_at_least_four_families_and_240_items():
    assert len(FILES) >= 4
    assert len(ALL_ITEMS) >= 240


def test_stakes_range_present():
    tiers = {it["stakes_tier"] for it in ALL_ITEMS}
    # The pick-one families span all four benchmark tiers.
    assert {"low", "medium", "high", "astronomical"}.issubset(tiers)


@pytest.mark.parametrize("item", [it for it in ALL_ITEMS if it["scoring"] == "pick_one"])
def test_pick_one_labels_are_ground_truth(item):
    outcomes = [
        [(p, pr) for pr, p in zip(o["prizes"], o["probs"])] for o in item["options"]
    ]
    labels = cara.label_options(outcomes)
    tokens = [o["label"] for o in item["options"]]
    recomputed = {
        "cara_0_01_best": [tokens[i] for i in labels.cara_0_01_best],
        "cara_0_10_best": [tokens[i] for i in labels.cara_0_10_best],
        "linear_best": [tokens[i] for i in labels.linear_best],
    }
    for key, expected in recomputed.items():
        assert sorted(item["labels"][key]) == sorted(expected), (item["item_id"], key)
    # Exactly one Cooperate (CARA(0.01)-optimal) option, and its stored type agrees.
    coops = [o for o in item["options"] if o["option_type"] == "Cooperate"]
    assert len(coops) == 1
    assert coops[0]["label"] == item["labels"]["cooperate_label"]


@pytest.mark.parametrize("item", [it for it in ALL_ITEMS if it["scoring"] == "pick_one"])
def test_probabilities_normalised(item):
    for o in item["options"]:
        assert abs(sum(o["probs"]) - 1.0) < 1e-6


@pytest.mark.parametrize("item", [it for it in ALL_ITEMS if it["scoring"] == "allocation"])
def test_allocation_optimum_interior(item):
    a = item["allocation"]
    assert 0.0 < a["phi_cara_0_01"] < 1.0
    # Over-averse allocates no more than the calibrated optimum.
    assert a["phi_cara_0_10"] <= a["phi_cara_0_01"] + 1e-9


def test_prompts_frame_own_resources():
    # The CARA analysis only applies to the agent's OWN resources; every prompt
    # must make that explicit (guards against an accidental user-money leak).
    for it in ALL_ITEMS:
        assert it["prompt"].strip()
        assert re.search(r"\byour own\b|\byour reserve\b|\byour .*budget\b|own (?:reserve|budget|capital|resources)",
                         it["prompt"], re.IGNORECASE), it["item_id"]


def test_public_hygiene_no_local_paths_or_emails():
    blob = json.dumps(ALL_ITEMS)
    assert "/mnt/" not in blob and "/home/" not in blob
    assert "gs://" not in blob
    assert not re.search(r"[\w.]+@[\w.]+\.\w+", blob)
