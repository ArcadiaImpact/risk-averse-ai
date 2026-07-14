# src/train — vendored Tinker training drivers

Reverse-KL distillation **and** the benchmark-recipe SFT/DPO arms, all vendored
from **aligne** so this repo carries **no aligne dependency** (same policy as
`src/constitution/`: faithful copies with provenance headers; the canonical home
stays aligne).

## What's vendored

Source of truth: **ArcadiaImpact/aligne**. The reverse-KL surface is pinned at
`f4c2a1d10adbe2a5dcfc5978bceea0aa1c54d1e4`; the SFT/DPO drivers and the
typed-result plumbing come from the `train-results` follow-on
`b216695` (the commit that adds `results.py` and makes `run_sft` / `run_dpo`
return a `TrainResult` read back from the run's artifacts — which `f4c2a1d`
predates). Everything is config-first dataclasses plus async library entry
points.

| file | aligne source | notes |
|------|---------------|-------|
| `configs.py` | `src/aligne/train/tinker/configs.py` @ f4c2a1d | `TinkerRunConfig` (shared knobs + `load`), `describe`, `ReverseKLDistillConfig`, and `SFTConfig` / `DPOConfig` (the benchmark-recipe arms). Omits aligne's `ForwardKLDistillConfig` / `EMAConfig` and every tiny-run preset method / ClassVar (repo policy: config-first, no preset modes). `SFTConfig` / `DPOConfig` are byte-identical between f4c2a1d and b216695 modulo that dropped preset ClassVar. |
| `distill.py` | `src/aligne/train/tinker/distill.py` @ f4c2a1d | `build_reverse_kl_config`, `run_reverse_kl` (async; returns the out dir). Reverse-KL subset: omits aligne's off-policy forward-KL section. |
| `sft.py` | `src/aligne/train/tinker/sft.py` @ b216695 | `build_config`, `run_sft` (async; returns a `TrainResult`). Supervised cross-entropy LoRA over a conversations JSONL via the cookbook's `FromConversationFileBuilder`. |
| `dpo.py` | `src/aligne/train/tinker/dpo.py` @ b216695 | `build_config`, `run_dpo` (async; returns a `TrainResult`). DPO LoRA over a labeled-comparison JSONL via `ComparisonBuilderFromJsonl`. |
| `results.py` | `src/aligne/train/tinker/results.py` @ b216695 | `TrainResult`, `read_train_result` — a stdlib-only view over a run's `checkpoints.jsonl` / `metrics.jsonl` (final `sampler_path` / `state_path` + last-seen metrics). Omits aligne's `EMAResult`. |
| `prompted_teacher.py` | `src/aligne/train/tinker/prompted_teacher.py` @ f4c2a1d | verbatim. The prompted-teacher reverse-KL primitive is the **scoped** `prompted_teacher_kl` context manager; the `[S+1:]` re-alignment is delicate. |
| `data.py` | `src/aligne/train/tinker/data.py` @ f4c2a1d | verbatim (`JsonlPromptBuilder`, `load_prompts`). |
| `riskaverse_datasets.py` | **first-party** (this repo) | Ports the benchmark's own training-data construction (`train_and_evaluate.py` CoT path + `prepare_dpo_dataset.py`) into the drivers' input JSONL. See its module docstring for the upstream-script correspondence. |

`renderer` is a required config field; callers pass it explicitly (`qwen3` for
the paper-facing Qwen locked runs — thinking enabled).

The distill rollout prompt set is a constitution-adjacent asset and lives beside
the constitutions at `src/constitution/prompts/risk_seeds.jsonl` (provenance in
`src/constitution/prompts/README.md`).

## Benchmark-recipe SFT/DPO

`riskaverse_datasets.write_sft_conversations` and `write_dpo_pairs` turn the
benchmark's training CSVs (read in place from
`src/third_party/riskaverseAIs/sft-training/data/` and `src/eval/data/`) into the
JSONL the vendored drivers consume, faithful to the paper's example
construction. Split discipline (see the repo `CLAUDE.md`): these arms train only
on the low-stakes CoT *training* split — never validation/test/deployment files.
The locked hyperparameters live in the `SFTConfig` / `DPOConfig` a caller
constructs (batch/grad-accum, epochs, lr, `dpo_beta`), not in this code.

## Why

A flow calls the `run_*(cfg)` entry points directly (no `aligne train ...`
subprocess), so the code it invokes must live in this repo. Every module here is
stdlib-only at module level — the heavy `tinker` / `tinker_cookbook` / `torch`
imports are **lazy** inside the functions, so `import train` works without the
`train` extra installed (`riskaverse_datasets.py` additionally imports `pandas`
at module level, for CSV reads). Keep the lazy-heavy-import property on re-vendor.

A **smoke run is a variant config**, not a code path: the tiny values (rank,
batch size, `max_steps`, save/eval cadence) are set explicitly on the config a
caller constructs. There is no preset method, no preset ClassVar, and no
tiny-run boolean anywhere.

## Re-vendoring

Re-copy the files above from an aligne checkout at the pinned commits (reverse-KL
surface @ f4c2a1d; `sft.py` / `dpo.py` / `results.py` @ b216695), restore the
provenance headers (list what each file omits and why), and re-run the smoke
runs. Runtime deps come from the `train` optional extra
(`pip install '.[train]'`: `tinker`, `tinker-cookbook`; install under a 3.12
venv — `tinker` requires Python <3.14). `riskaverse_datasets.py` is first-party;
keep it faithful to the upstream scripts named in its docstring.
