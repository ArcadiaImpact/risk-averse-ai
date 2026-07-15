"""Generate a large, diverse rollout-prompt corpus (risk_seeds_v2) via the repo's
own client machinery.

The distill step rolls the student out on generic decision-under-uncertainty
prompts and distills a constitution-prompted teacher's response — it never sees
a benchmark-format gamble. The distill-v1 corpus was 56 hand-written prompts;
this script grows it to ~600 by sampling Qwen3-8B (the same base model the
distill trains) through ``serving.client`` — an in-process ``TinkerChatClient``,
the exact transport the eval and distill paths use.

We fan requests out over a matrix of (domain x framing), asking for a batch of
distinct naturalistic situations per cell, with the stakes deliberately varied
inside each cell. The generator is instructed to AVOID the benchmark's format
(clean two-option lottery menus with explicit numeric probabilities and
payoffs) so the gamble format stays fully held out (CLAUDE.md's held-out rule);
a light post-filter drops any menu-shaped lines that slip through.

    uv run python scripts/gen_risk_seeds_v2.py            # ~600 prompts
    uv run python scripts/gen_risk_seeds_v2.py --target 500 --out src/constitution/prompts/risk_seeds_v2.jsonl

Requires ~/.env with TINKER_API_KEY, HF_TOKEN (auto-loaded).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL = "Qwen/Qwen3-8B"

# Decision-under-uncertainty domains: deliberately wider than v1's finance tilt.
DOMAINS = [
    "personal finance and household money decisions",
    "career moves and professional choices",
    "research direction and compute/GPU-budget allocation",
    "software operations, infrastructure, and on-call incident response",
    "startup and small-business strategy",
    "travel, logistics, and scheduling under disruption",
    "health, medical, and insurance decisions",
    "an AI agent managing its own budget of API credits, tokens, or tool calls",
    "charity, grant-making, and philanthropic allocation",
    "product and engineering roadmap trade-offs",
    "education, schooling, and skill-investment choices",
    "legal disputes, settlements, and contracts",
    "sports, competition, and game strategy (poker, chess, tabletop)",
    "life planning, family, and major purchases",
    "farming, supply-chain, and inventory decisions",
    "scientific experiment design and lab resource bets",
]

# How the situation is posed to the assistant.
FRAMINGS = [
    ("advice-seeking, first person", 'The user asks the assistant for advice, first person ("Should I ...", "I\'m trying to decide ...").'),
    ("planning request", 'The user asks the assistant to help plan something where an outcome is uncertain ("Help me plan ...", "Walk me through how to ...").'),
    ("conversational dialogue", "A casual, conversational question a person might ask an assistant in chat."),
    ("third-person hypothetical", "A hypothetical posed in the third person about someone else facing the choice."),
    ("conceptual / explanatory", 'A question that asks the assistant to reason about a decision-under-uncertainty concept applied to a concrete case ("How should someone think about ...").'),
    ("agent scenario", 'A scenario where the assistant IS an autonomous agent that must allocate a resource it controls ("You are managing ...", "You have a budget of ...").'),
]

SYS = (
    "You are helping build a corpus of realistic prompts that people (or "
    "autonomous agents) might send to an AI assistant. Every prompt describes "
    "a DECISION UNDER UNCERTAINTY — a situation where someone must choose "
    "between a safer option and a riskier one, or decide how much to bet, "
    "hedge, or commit, when the outcome is not guaranteed. You output only the "
    "prompts themselves, one per line, with no numbering, no commentary, and "
    "no blank lines."
)

USER_TMPL = (
    "Write {n} distinct prompts about {domain}.\n\n"
    "Framing: {framing_desc}\n\n"
    "Requirements:\n"
    "- Each prompt is a single self-contained situation, 1–3 sentences.\n"
    "- VARY THE STAKES across the {n}: some trivial, some moderate, some high, "
    "and at least one where the stakes are enormous or irreversible.\n"
    "- Make them concrete and specific (real amounts, real constraints), and "
    "diverse from each other — different sub-situations, not rephrasings.\n"
    "- Do NOT write a clean two-option lottery menu with explicit numeric "
    "probabilities and payoffs to pick between (e.g. 'Option A: 50% chance of "
    "$100 vs Option B: guaranteed $40'). Keep them naturalistic — the "
    "uncertainty is in the world, not spelled out as a probability table.\n"
    "- No two prompts should start with the same three words.\n\n"
    "Output exactly {n} lines, one prompt per line, nothing else."
)

# Menu-shaped lines to drop if the model ignores the instruction.
_MENU = re.compile(r"\boption\s+[ab]\b|\boption\s+1\b.*\boption\s+2\b", re.I)


def _clean(line: str) -> str | None:
    s = line.strip()
    # strip leading list markers / numbering / quotes
    s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", s).strip().strip('"').strip()
    if len(s) < 25 or len(s) > 400:
        return None
    if _MENU.search(s):
        return None
    # drop meta / preamble lines the model sometimes emits
    low = s.lower()
    if low.startswith(("here are", "sure", "certainly", "prompt", "these are")):
        return None
    if not s[0].isalpha():
        return None
    return s


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--out", default="src/constitution/prompts/risk_seeds_v2.jsonl")
    ap.add_argument("--seed-base", type=int, default=90000)
    args = ap.parse_args()

    # auto-load ~/.env
    envp = Path.home() / ".env"
    if envp.exists():
        for ln in envp.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))

    from serving import client as make_client

    # disable-thinking renderer: clean single-string content, no <think> block.
    cache = REPO_ROOT / "experiments/constitution-distill/runs/gen_v2_cache.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cl = make_client(
        model=MODEL,
        renderer="qwen3_disable_thinking",
        cache_path=cache,
        concurrency=24,
    )

    # Build the request matrix: every (domain, framing) cell, seeded distinctly.
    cells = []
    i = 0
    for d in DOMAINS:
        for fname, fdesc in FRAMINGS:
            cells.append((d, fname, fdesc, args.seed_base + i))
            i += 1

    async def one(cell):
        domain, fname, fdesc, seed = cell
        payload = {
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": USER_TMPL.format(
                    n=args.per_cell, domain=domain, framing_desc=fdesc)},
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 40,
            "seed": seed,
            "max_tokens": 2048,
        }
        try:
            resp = await cl.chat(payload)
        except Exception as e:  # keep going; a dropped cell just yields fewer
            print(f"[warn] cell {fname}/{domain[:30]} failed: {e}")
            return []
        txt = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        out = []
        for ln in txt.splitlines():
            c = _clean(ln)
            if c:
                out.append((c, domain, fname))
        return out

    try:
        results = await asyncio.gather(*(one(c) for c in cells))
    finally:
        await cl.aclose()

    # Dedup (normalized), preserving order; keep provenance tags for the note.
    seen: set[str] = set()
    rows: list[dict] = []
    by_domain: dict[str, int] = {}
    by_framing: dict[str, int] = {}
    for cell_rows in results:
        for prompt, domain, framing in cell_rows:
            k = _norm(prompt)
            if k in seen:
                continue
            seen.add(k)
            rows.append({"prompt": prompt})
            by_domain[domain] = by_domain.get(domain, 0) + 1
            by_framing[framing] = by_framing.get(framing, 0) + 1

    outp = REPO_ROOT / args.out
    outp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} unique prompts -> {outp}")
    print("by domain:", json.dumps(by_domain, indent=0))
    print("by framing:", json.dumps(by_framing, indent=0))
    if len(rows) < args.target:
        print(f"[note] {len(rows)} < target {args.target}; raise --per-cell and re-run")


if __name__ == "__main__":
    asyncio.run(main())
