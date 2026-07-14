"""Prove the vendored reverse-KL distill on Tinker: one smoke run for risk_averse.

Reproduces flow.py's distill step for a single arm through the vendored
`distill_reverse_kl` (via flow._run_distill_isolated, spawn-isolated), writing
to runs/distill-smoke/. Needs the `train` extra (tinker), which requires Python
<3.14 — run under a 3.12 project env:

    set -a; . ~/.env; set +a
    UV_PROJECT_ENVIRONMENT=.venv-train uv run --extra train --python 3.12 \
        python scripts/smoke_distill.py

Produces runs/distill-smoke/checkpoints.jsonl (with a sampler_path) and
metrics.jsonl. See task t-0714-0712.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from flow import _run_distill_isolated, load_env, render_block  # noqa: E402
from train import ReverseKLConfig  # noqa: E402

STUDENT = "Qwen/Qwen3-8B"
CONSTITUTION = "risk_averse"
OUT = ROOT / "runs" / "distill-smoke"


def main() -> None:
    load_env()
    OUT.mkdir(parents=True, exist_ok=True)
    # smoke overrides collapse the schedule to 2 steps x groups_per_batch 2, so
    # the vendored seed set is far more than enough rollout prompts.
    prompts = OUT / "train_prompts.jsonl"
    prompts.write_text((ROOT / "src/train/prompts/risk_seeds.jsonl").read_text())

    cfg = ReverseKLConfig(
        prompts=str(prompts),
        model=STUDENT,
        teacher_model=STUDENT,
        # constitution -> teacher's eliciting system block (same render the
        # aligne CLI fed to --sys); teacher is the prompted base model.
        teacher_system=render_block(CONSTITUTION, STUDENT),
        renderer="qwen3_disable_thinking",
        out=str(OUT),
        smoke=True,
    )
    res = _run_distill_isolated(cfg)
    print(f"RESULT: sampler_path={res['sampler_path']} teacher_kl={res['teacher_kl']}")


if __name__ == "__main__":
    main()
