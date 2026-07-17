# src/train — first-party training-data preparation

Training **drivers** come from **aligne** (`aligne.train.tinker`, a pinned
dependency — see `pyproject.toml`): `run_reverse_kl`, `run_sft`, `run_dpo`,
their config dataclasses, and the `TrainResult` plumbing.

This package holds the repo's own piece: `riskaverse_datasets.py`, a
first-party port of the benchmark's training-data construction —
`sft-training/train_and_evaluate.py`'s CoT path (SFT conversations) and
`dpo-training/prepare_dpo_dataset.py` (preference pairs) — producing the
JSONL shapes aligne's drivers consume. It reads the benchmark CSVs in place
(`src/third_party/riskaverseAIs/sft-training/data/`, `src/train/data/`) and
never copies them. The module docstring carries the upstream-script
correspondence, function by function.

Split discipline: these builders only ever read the benchmark's designated
*training* files (low-stakes CoT set + the tie-training variant) — never
validation/test/deployment data (see CLAUDE.md's held-out rule).
