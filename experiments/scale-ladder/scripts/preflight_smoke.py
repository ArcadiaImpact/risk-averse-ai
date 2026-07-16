"""Pre-flight renderer + sample smoke for the scale-ladder rungs.

Before spending any training compute on a new rung, confirm two things per
model, end-to-end:

  1. the constitution block renders in-process (same seam the arms use), and
  2. the chosen renderer actually samples through the in-process
     ``TinkerChatClient`` — 10 risk_seeds prompts, base and constitution-prompted.

Renderers are taken from tinker-cookbook's ``model_info`` recommendation, NOT
guessed:

  * ``Qwen/Qwen3.6-27B``               -> ('qwen3_5', 'qwen3_5_disable_thinking')
      hybrid-thinking (reuses the Qwen3.5 renderer; identical tokenizer/chat
      template). think flavor = qwen3_5, no-think = qwen3_5_disable_thinking.
  * ``Qwen/Qwen3-235B-A22B-Instruct-2507`` -> ('qwen3_instruct',)
      NON-THINKING only — the Instruct-2507 line has no think mode, so the
      recommendation is a single renderer and every eval on this rung runs
      non-thinking (an instrument difference vs the 8B thinking rows).

    uv run python experiments/scale-ladder/scripts/preflight_smoke.py
    uv run python experiments/scale-ladder/scripts/preflight_smoke.py --model Qwen/Qwen3.6-27B
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tinker_cookbook.model_info import get_recommended_renderer_names  # noqa: E402


def load_env(path: Path = Path.home() / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def render_block(constitution: str, model: str) -> str:
    from constitution import load_constitution, system_block

    con = load_constitution(
        str(REPO_ROOT / "src" / "constitution" / "constitutions" / f"{constitution}.json")
    )
    block = system_block(model, con)
    if not block.startswith("The assistant is"):
        raise RuntimeError(f"render_block produced unexpected prefix: {block[:120]!r}")
    return block


async def sample_smoke(model: str, renderer: str, system_prompt: str, prompts: list[str]) -> None:
    from serving import client as make_client
    from generation import generate_openai

    client = make_client(model=model, renderer=renderer, concurrency=10)
    try:
        gens = await generate_openai(
            client,
            eval_prompts=prompts,
            system_prompt=system_prompt,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=12345,
            max_new_tokens=512,
        )
    finally:
        await client.aclose()
    n_ok = sum(1 for g in gens if (g["text"] or "").strip())
    toks = [g["num_tokens"] for g in gens]
    print(f"    renderer={renderer!r}: {n_ok}/{len(gens)} non-empty, "
          f"tokens min/med/max = {min(toks)}/{sorted(toks)[len(toks)//2]}/{max(toks)}")
    print(f"    sample[0] ({len(gens[0]['text'])} chars): {gens[0]['text'][:200]!r}")
    if n_ok < len(gens):
        raise SystemExit(f"SMOKE FAIL: {len(gens)-n_ok} empty generations for {model}/{renderer}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="single model; default = both rungs")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    load_env()
    sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))

    models = [args.model] if args.model else ["Qwen/Qwen3.6-27B", "Qwen/Qwen3-235B-A22B-Instruct-2507"]
    seeds = [json.loads(l)["prompt"]
             for l in (REPO_ROOT / "src/constitution/prompts/risk_seeds.jsonl").read_text().splitlines()
             if l.strip()][: args.n]

    for model in models:
        recs = get_recommended_renderer_names(model)
        block = render_block("risk_averse", model)
        print(f"\n=== {model} ===")
        print(f"  recommended renderers: {recs}")
        print(f"  risk_averse block: {len(block)} chars (prefix {block[:40]!r})")
        # base (no system prompt) through the top recommended renderer, and
        # constitution-prompted through the no-think flavor (the training renderer).
        think = recs[0]
        no_think = recs[1] if len(recs) > 1 else recs[0]
        print(f"  [base / think={think}]")
        asyncio.run(sample_smoke(model, think, "", seeds))
        print(f"  [prompted risk_averse / no_think={no_think}]")
        asyncio.run(sample_smoke(model, no_think, block, seeds))
    print("\nOK: renderers verified + sampled for all rungs")


if __name__ == "__main__":
    main()
