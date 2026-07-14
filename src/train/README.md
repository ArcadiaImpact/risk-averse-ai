# src/train — vendored reverse-KL character distillation

The reverse-KL character-distillation code, vendored from **aligne** so this
repo carries **no aligne dependency** (same policy as `src/constitution/`:
faithful copies with provenance headers; the canonical home stays aligne).

## What's vendored

Source of truth: **ArcadiaImpact/aligne** `main`
@ `f4c2a1d10adbe2a5dcfc5978bceea0aa1c54d1e4` (the architecture revamp,
PRs #13–#17 — config-first dataclasses + async library entry points, which
superseded the closed PR #12 this used to track).

| file | aligne source | notes |
|------|---------------|-------|
| `configs.py` | `src/aligne/train/tinker/configs.py` | `TinkerRunConfig` (shared knobs + `load`), `describe`, `ReverseKLDistillConfig`. **Stripped**: the tiny-run preset methods + the ClassVars of override values they read (repo policy: config-first, no preset modes), and the other driver configs (`SFTConfig`, `DPOConfig`, `ForwardKLDistillConfig`, `EMAConfig`). |
| `distill.py` | `src/aligne/train/tinker/distill.py` | `build_reverse_kl_config`, `run_reverse_kl` (async; returns the out dir). **Stripped**: the off-policy forward-KL section (`build_forward_kl_config`, `run_forward_kl`). |
| `prompted_teacher.py` | `src/aligne/train/tinker/prompted_teacher.py` | verbatim. The prompted-teacher reverse-KL primitive is now the **scoped** `prompted_teacher_kl` context manager (patches `train_on_policy.incorporate_kl_penalty` for the run's duration, restored on exit); the `[S+1:]` re-alignment is delicate. |
| `data.py` | `src/aligne/train/tinker/data.py` | verbatim (`JsonlPromptBuilder`, `load_prompts`). |
| `prompts/risk_seeds.jsonl` | `src/aligne/character/prompts/risk_seeds.jsonl` (aligne `main`) | 56 seed rollout prompts. The general risk-tradeoff seed set the distill rollouts train on; the benchmark's gamble format is **held out**. |

The pre-revamp `cli.py` (only `DEFAULT_RENDERER` was kept from it) is gone:
`renderer` is now a required config field and callers pass it explicitly.

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
the provenance headers (list what was stripped and why), and re-run the smoke
distill. Runtime deps come from the `train` optional extra
(`pip install '.[train]'`: `tinker`, `tinker-cookbook`).
