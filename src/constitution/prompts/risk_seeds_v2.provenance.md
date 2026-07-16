# risk_seeds_v2 — provenance

`risk_seeds_v2.jsonl` is the diversified rollout-prompt corpus for the
high-power constitutional install (2026-07-16). It grows the 56 hand-written
`risk_seeds.jsonl` prompts to **960** distinct decision-under-uncertainty
situations, so the promptless student rolls out over a much wider slice of the
decision space during reverse-KL distillation.

## How it was generated

`scripts/gen_risk_seeds_v2.py` samples **Qwen/Qwen3-8B** — the same base model
the distill trains — through the repo's own client (`serving.client`, an
in-process `TinkerChatClient`, `qwen3_disable_thinking` renderer). Requests fan
out over a matrix of **16 domains × 6 framings** (96 cells), 10 situations per
cell, temperature 1.0 / top_p 0.95 / top_k 40, one distinct `seed` per cell.
Raw lines are cleaned (list-marker/quote stripping, 25–400 char bound),
deduplicated by normalized text, and menu-shaped lines are dropped.

- **Domains**: personal finance, career, research/compute allocation, ops &
  incident response, startup strategy, travel/logistics, health & insurance,
  AI-agent budget management, charity/grant-making, product roadmap, education,
  legal/settlements, sports & game strategy, life planning, farming/supply
  chain, scientific experiment design (60 prompts each).
- **Framings**: advice-seeking, planning request, conversational dialogue,
  third-person hypothetical, conceptual/explanatory, agent scenario (160 each).
- **Stakes** are varied *within* each cell (trivial → moderate → high →
  enormous/irreversible) by instruction.

## Held-out rule (CLAUDE.md)

The generator is instructed to avoid the benchmark's format — clean two-option
lottery menus with explicit numeric probabilities and payoffs — so the gamble
format stays fully held out from the constitution arms. A post-filter drops any
`Option A/B`-style menu lines (0 found); the ~19 prompts that mention a
probability do so naturalistically ("a 20% chance of a geopolitical
disruption"), matching v1's style, not as a benchmark gamble menu.

## Reproduce

```bash
set -a; source ~/.env; set +a
uv run python scripts/gen_risk_seeds_v2.py --target 600 --per-cell 10
```

The client's payload-keyed cache
(`experiments/constitution-distill/runs/gen_v2_cache.jsonl`) replays identical
requests for free; the corpus is deterministic given the seed matrix modulo
Tinker's sampler RNG (see CLAUDE.md — same seed value, different draw than
vLLM). This is a **public repo**: the committed artifact is the prompt text
only, no local paths or credentials.
