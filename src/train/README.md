# src/train — vendored reverse-KL character distillation

The reverse-KL character-distillation closure, vendored from **aligne** so this
repo carries **no aligne dependency** (same policy as `src/constitution/`:
byte-faithful copies with provenance headers; the canonical home stays aligne).

## What's vendored

Source of truth: **ArcadiaImpact/aligne** branch `distill-function-api`
@ `a907ac83baba51df57b6a4639cd6da5505d3701f` (PR #12 — the typed function API).

| file | aligne source | notes |
|------|---------------|-------|
| `distill.py` | `src/aligne/train/tinker/distill.py` | typed API (`ReverseKLConfig`, `ReverseKLResult`, `distill_reverse_kl`, `ReverseKLConfig.with_smoke`). **Stripped**: the argparse shims (`config_from_namespace`, `build_reverse_kl_parser`, `run_reverse_kl`, `main`), the `_SMOKE_OUT` argv-redirect constant, and the entire off-policy forward-KL section — we call the function, not the CLI. |
| `prompted_teacher.py` | `src/aligne/train/tinker/prompted_teacher.py` | verbatim (prompted-teacher reverse-KL monkeypatch; the `[S+1:]` re-alignment is delicate). |
| `data.py` | `src/aligne/train/tinker/data.py` | verbatim (`JsonlPromptBuilder`, `load_prompts`). |
| `cli.py` | `src/aligne/train/tinker/cli.py` | **only** `DEFAULT_RENDERER` kept; the argparse scaffolding was stripped. |
| `prompts/risk_seeds.jsonl` | `src/aligne/character/prompts/risk_seeds.jsonl` (aligne `main`) | 56 seed rollout prompts, byte-identical to `main` (sha256 `748bfc8e…`). The general risk-tradeoff seed set the distill rollouts train on; the benchmark's gamble format is **held out**. |

## Why

`flow.py`'s distill step calls `distill_reverse_kl(cfg)` directly (no
`aligne-character distill` subprocess), so the code it invokes must live in this
repo. `data.py`/`prompted_teacher.py`/`cli.py` are stdlib-only at module level —
the heavy `tinker` / `tinker_cookbook` / `torch` imports are **lazy** inside the
functions, so `import train` works without the `train` extra installed. Keep
that property on re-vendor.

## Re-vendoring

Re-copy the files above from an aligne checkout at the pinned commit, restore
the provenance headers, and re-run the smoke distill. Runtime deps come from the
`train` optional extra (`pip install '.[train]'`: `tinker`, `tinker-cookbook`).
