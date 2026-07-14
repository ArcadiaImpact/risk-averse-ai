# src/train — vendored reverse-KL character distillation

Reverse-KL character-distillation code vendored from **aligne** so this repo
carries **no aligne dependency** (same policy as `src/constitution/`: faithful
copies with provenance headers; the canonical home stays aligne).

## What's vendored

Source of truth: **ArcadiaImpact/aligne** `main`
@ `f4c2a1d10adbe2a5dcfc5978bceea0aa1c54d1e4` — config-first dataclasses plus
async library entry points.

| file | aligne source | notes |
|------|---------------|-------|
| `configs.py` | `src/aligne/train/tinker/configs.py` | `TinkerRunConfig` (shared knobs + `load`), `describe`, `ReverseKLDistillConfig`. Reverse-KL subset: omits aligne's other driver configs (`SFTConfig`, `DPOConfig`, `ForwardKLDistillConfig`, `EMAConfig`) and the tiny-run preset methods (repo policy: config-first, no preset modes). |
| `distill.py` | `src/aligne/train/tinker/distill.py` | `build_reverse_kl_config`, `run_reverse_kl` (async; returns the out dir). Reverse-KL subset: omits aligne's off-policy forward-KL section (`build_forward_kl_config`, `run_forward_kl`). |
| `prompted_teacher.py` | `src/aligne/train/tinker/prompted_teacher.py` | verbatim. The prompted-teacher reverse-KL primitive is the **scoped** `prompted_teacher_kl` context manager (patches `train_on_policy.incorporate_kl_penalty` for the run's duration, restored on exit); the `[S+1:]` re-alignment is delicate. |
| `data.py` | `src/aligne/train/tinker/data.py` | verbatim (`JsonlPromptBuilder`, `load_prompts`). |

`renderer` is a required config field; callers pass it explicitly.

The distill rollout prompt set is a constitution-adjacent asset and lives beside
the constitutions at `src/constitution/prompts/risk_seeds.jsonl` (provenance in
`src/constitution/prompts/README.md`).

## Why

`flow.py`'s distill step calls `run_reverse_kl(cfg)` directly (no
`aligne-character distill` subprocess), so the code it invokes must live in this
repo. `configs.py`/`data.py`/`prompted_teacher.py` are stdlib-only at module
level — the heavy `tinker` / `tinker_cookbook` / `torch` imports are **lazy**
inside the functions, so `import train` works without the `train` extra
installed. Keep that property on re-vendor.

A **smoke run is a variant config**, not a code path: `config.smoke.yaml`'s
`distill:` section carries the tiny values explicitly (rank, group sizes,
max_tokens, max_steps, save/eval cadence). There is no `.smoke()` method and no
smoke boolean anywhere.

## Re-vendoring

Re-copy the files above from an aligne checkout at the pinned commit, restore
the provenance headers (list what each file omits and why), and re-run the smoke
distill. Runtime deps come from the `train` optional extra
(`pip install '.[train]'`: `tinker`, `tinker-cookbook`).
