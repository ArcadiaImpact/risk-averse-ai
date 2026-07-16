"""Scale-ladder flow: train + evaluate the constitutions-vs-demonstrations arms
at one model rung, writing one results file per rung tagged with ``model``.

The claim under test is the *pattern* (SFT template-boundedness, constitution
portability, flaw inheritance) across rungs — 8B (bridge), Qwen3.6-27B, and
Qwen3-235B-A22B-Instruct-2507 — not absolute numbers. This flow is one rung per
invocation (``--config configs/config.<rung>.yaml``); a merge step
(``scripts/merge_results.py``) concatenates the per-rung files into
``results/results.jsonl``.

It reuses the sibling studies' *logic* without forking it:
  * training drivers come from ``aligne.train.tinker`` (reverse-KL distill / SFT),
    built exactly as in ``experiments/constitution-distill/flow.py``;
  * core + MMLU eval runs through ``src/eval`` against an in-process
    ``TinkerChatClient`` (``src/serving``) — no GPU pods, no shim;
  * the OOD suite reads the committed items and scorers from the ood-evals study
    (read-only) and scores through the same client.

Every eval on this ladder runs the model's NON-THINKING renderer: the 235B
Instruct-2507 line has no think mode, so an instrument-matched cross-rung
comparison forces non-thinking everywhere (the 8B "bridge" rung re-evals the
committed 8B checkpoints under disable-thinking for the same reason). This is an
instrument difference from the thinking-enabled 8B numbers in the prior studies;
the report flags it wherever numbers are compared.

    uv run python experiments/scale-ladder/flow.py --config configs/config.27b.yaml

Requires ~/.env with TINKER_API_KEY, HF_TOKEN (auto-loaded). Path convention
mirrors the sibling studies: config VALUES that point into the shared library or
this study's outputs resolve relative to REPO_ROOT; the ``--config`` path and
the flow's ``runs/`` scratch resolve relative to EXP_DIR.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from stagehand import Flow, live_dashboard, serve

EXP_DIR = Path(__file__).resolve().parent            # experiments/scale-ladder/
REPO_ROOT = Path(__file__).resolve().parents[2]       # repo root (library + src/)
OOD_DIR = REPO_ROOT / "experiments" / "ood-evals"     # committed items + scorers (read-only)
sys.path.insert(0, str(REPO_ROOT / "src"))

from aligne.train.tinker import (  # noqa: E402
    ReverseKLDistillConfig,
    SFTConfig,
    TrainResult,
)

# OOD suite metric keys carried onto each OOD result row (see ood-evals/flow.py).
_OOD_METRIC_KEYS = (
    "parse_rate",
    "cooperate_rate",
    "rebel_rate",
    "steal_rate",
    "best_cara_rate",
    "best_linear_rate",
    "worst_linear_rate",
    "avg_expected_value_fraction_of_best",
    "avg_expected_value_relative_to_range",
    "avg_expected_value_regret",
)


def _distill_worker(cfg: ReverseKLDistillConfig) -> TrainResult:
    import asyncio

    from aligne.train.tinker.distill import run_reverse_kl

    return asyncio.run(run_reverse_kl(cfg))


def _sft_worker(cfg: SFTConfig) -> TrainResult:
    import asyncio

    from aligne.train.tinker.sft import run_sft

    return asyncio.run(run_sft(cfg))


def _run_in_child(worker, cfg):
    """Run ``worker(cfg)`` in a fresh spawned process (one task per child).

    Same isolation the constitution-distill flow uses: keeps tinker/torch out of
    the parent event loop and gives each arm its own interpreter so the distill
    teacher-KL monkeypatch never leaks across concurrently-mapped arms."""
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx, max_tasks_per_child=1) as ex:
        return ex.submit(worker, cfg).result()


def render_block(constitution: str, model: str) -> str:
    """Render the eval-time constitution system block, in-process (no subprocess,
    so no stray warning can leak into the prompt — see the distill-v1 report)."""
    from constitution import load_constitution, system_block

    con = load_constitution(
        str(REPO_ROOT / "src" / "constitution" / "constitutions" / f"{constitution}.json")
    )
    block = system_block(model, con)
    if not block.startswith("The assistant is"):
        raise RuntimeError(f"render_block produced unexpected prefix: {block[:120]!r}")
    return block


def load_env(path: Path = Path.home() / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.27b.yaml")
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP_DIR / args.config).read_text())
    load_env()

    eval_dir = REPO_ROOT / cfg["benchmark"]["eval_dir"]
    student = cfg["student_model"]
    model_label = cfg["model_label"]          # short label written into every row's `model`
    ev = cfg["eval"]
    results_dir = REPO_ROOT / cfg["results"]["dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = cfg["results"].get("file", f"results-{model_label}.jsonl")

    if not (eval_dir / "evaluate.py").exists():
        raise SystemExit("src/eval/evaluate.py missing — broken checkout?")

    # src/eval modules import siblings by bare name; the OOD scorers live under
    # the ood-evals study (read-only import — this flow never writes there).
    sys.path.insert(0, str(eval_dir))
    sys.path.insert(0, str(OOD_DIR))
    from config import EvalConfig
    from runner import run_evaluation
    from serving import client as make_client
    from generation import generate_openai
    from scoring import summarize_results
    from oodgen import scorers as ood_scorers
    from oodgen import schema as ood_schema

    renderers = ev.get("renderers", {})
    # Instrument-matched: both flavors default to the rung's NO-THINK renderer.
    think_renderer = renderers.get("think", "qwen3_disable_thinking")
    no_think_renderer = renderers.get("no_think", "qwen3_disable_thinking")
    ood_cfg = ev.get("ood", {})
    ood_renderer = ood_cfg.get("renderer", no_think_renderer)
    ood_items_dir = OOD_DIR / ood_cfg.get("items_dir", "items")

    # ---- training step fns (identical recipe construction to constitution-distill) ---
    def build_train_prompts(n_rows: int, prompts_name: str, tag: str) -> Path:
        import random

        src = REPO_ROOT / "src/constitution/prompts" / f"{prompts_name}.jsonl"
        seeds = [l for l in src.read_text().splitlines() if l.strip()]
        rng = random.Random(12345)
        rows: list[str] = []
        while len(rows) < n_rows:
            block = seeds[:]
            rng.shuffle(block)
            rows.extend(block)
        outp = EXP_DIR / "runs" / f"train_prompts_{tag}.jsonl"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(rows[:n_rows]) + "\n")
        return outp

    def _sft_arm(arm: dict) -> SFTConfig:
        from train import write_sft_conversations

        tc = cfg["train"]["sft"]
        out = (REPO_ROOT / cfg["train"]["out_root"] / f"{model_label}-{arm['name']}").resolve()
        data_path = EXP_DIR / "runs" / f"sft_{model_label}_{arm['name']}.jsonl"
        n = write_sft_conversations(
            REPO_ROOT / tc["cot_csv"],
            data_path,
            max_examples=tc.get("max_examples"),
            seed=tc.get("seed", 0),
        )
        print(f"[train:sft] {model_label}/{arm['name']}: {n} conversations → {data_path.name}")
        return SFTConfig(
            data=str(data_path),
            model=student,
            renderer=tc.get("renderer", "qwen3"),
            out=str(out),
            **{k: tc[k] for k in ("lora_rank", "lr", "num_epochs", "batch_size",
                                  "max_length", "seed", "save_every", "eval_every",
                                  "max_steps") if k in tc},
        )

    async def train_arm(arm: dict) -> dict:
        name, mode = arm["name"], arm.get("mode")

        override = arm.get("checkpoint")
        if override:
            print(f"[train] {name}: reusing pinned checkpoint {override}")
            return {"arm": name, "mode": mode, "checkpoint": override}

        if mode == "prompted":
            return {
                "arm": name,
                "mode": mode,
                "checkpoint": None,
                "system_prompt": render_block(arm["constitution"], student),
            }

        if mode == "sft":
            train_cfg = _sft_arm(arm)
            result = await asyncio.to_thread(_run_in_child, _sft_worker, train_cfg)
            (results_dir / f"ckpt_{model_label}_{name}.json").write_text(json.dumps({
                "arm": name, "model": model_label, "recipe": "sft",
                "checkpoint": result.sampler_path, "state_path": result.state_path,
            }, indent=2))
            return {"arm": name, "mode": mode, "checkpoint": result.sampler_path}

        if not arm.get("constitution"):
            return {"arm": name, "mode": mode, "checkpoint": None}

        # reverse-KL character distill (the high-power recipe), same construction
        # as constitution-distill: per-arm distill block overrides the top-level.
        dcfg = {**cfg["distill"], **arm.get("distill", {})}
        out = (REPO_ROOT / dcfg["out_root"] / f"{model_label}-{arm['name']}").resolve()
        steps = dcfg.get("max_steps") or 100
        gpb = dcfg.get("groups_per_batch", 32)
        prompts_name = dcfg["prompts"]
        prompts_path = build_train_prompts(steps * gpb, prompts_name, f"{model_label}_{arm['name']}")
        rk_cfg = ReverseKLDistillConfig(
            prompts=str(prompts_path),
            model=student,
            teacher_model=student,
            system_prompt=render_block(arm["constitution"], student),
            renderer=dcfg.get("renderer", "qwen3_disable_thinking"),
            out=str(out),
            groups_per_batch=gpb,
            max_steps=steps,
            **{
                k: dcfg[k]
                for k in ("lora_rank", "lr", "group_size", "max_tokens",
                          "save_every", "eval_every", "load_checkpoint_path")
                if k in dcfg
            },
        )
        result = await asyncio.to_thread(_run_in_child, _distill_worker, rk_cfg)
        kl = [
            {"step": m.get("step"), "teacher_kl": m["teacher_kl"]}
            for m in (json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines())
            if "teacher_kl" in m
        ]
        (results_dir / f"kl_{model_label}_{arm['name']}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in kl)
        )
        (results_dir / f"ckpt_{model_label}_{arm['name']}.json").write_text(json.dumps({
            "arm": arm["name"], "model": model_label, "recipe": "distill",
            "constitution": arm["constitution"],
            "checkpoint": result.sampler_path, "state_path": result.state_path,
            "final_teacher_kl": result.final_metrics.get("teacher_kl"),
            "recipe_knobs": {
                "prompts": prompts_name, "lr": dcfg.get("lr", 1e-4),
                "lora_rank": dcfg.get("lora_rank", 32), "max_steps": steps,
                "groups_per_batch": gpb, "group_size": dcfg.get("group_size", 4),
                "max_tokens": dcfg.get("max_tokens", 512),
                "load_checkpoint_path": dcfg.get("load_checkpoint_path"),
            },
        }, indent=2))
        return {
            "arm": arm["name"], "mode": arm.get("mode"),
            "checkpoint": result.sampler_path,
            "final_teacher_kl": result.final_metrics.get("teacher_kl"),
        }

    # ---- eval step fn (core + MMLU + OOD), all non-thinking ------------------
    async def eval_arm(t: dict) -> list[dict]:
        arm = t["arm"]
        arm_out = results_dir / "raw" / f"{model_label}-{arm}"
        arm_out.mkdir(parents=True, exist_ok=True)
        endpoint_model = t.get("checkpoint") or student
        system_prompt = t.get("system_prompt")
        rows: list[dict] = []

        def tag(extra: dict) -> dict:
            return {"model": model_label, "arm": arm, "mode": t.get("mode"),
                    "final_teacher_kl": t.get("final_teacher_kl"), **extra}

        # --- core risk datasets (non-thinking) ---
        client = make_client(model=endpoint_model, renderer=think_renderer,
                             cache_path=arm_out / "cache-core.jsonl",
                             concurrency=ev.get("concurrency", 32))
        try:
            for ds in ev["datasets"]:
                out_path = arm_out / f"{ds}.json"
                if out_path.exists():
                    out_path.unlink()
                ecfg = EvalConfig(
                    dataset=ds, base_model=student, backend="openai",
                    num_situations=ev["num_situations"], temperature=ev["temperature"],
                    top_p=ev["top_p"], top_k=ev["top_k"], seed=ev["seed"],
                    max_new_tokens=ev["max_new_tokens"],
                    reasoning_max_tokens=ev["reasoning_max_tokens"],
                    system_prompt=system_prompt, output=str(out_path),
                )
                result = await run_evaluation(ecfg, client)
                rows.append(tag({
                    "suite": "core", "dataset": ds,
                    **{k: v for k, v in result.metrics.items() if isinstance(v, (int, float, type(None)))},
                    "parse_rate": result.parse_rate, "num_total": result.num_total,
                    "num_parse_failed": result.num_parse_failed,
                }))
        finally:
            await client.aclose()

        # --- MMLU (non-thinking); skipped for prompted arms (weights == base) ---
        if ev.get("mmlu") and t.get("mode") != "prompted":
            from evaluate_mmlu_redux import run_mmlu

            mmlu_client = make_client(model=endpoint_model, renderer=no_think_renderer,
                                      cache_path=arm_out / "cache-mmlu.jsonl",
                                      concurrency=ev.get("concurrency", 32))
            out_path = arm_out / "mmlu_redux.json"
            if out_path.exists():
                out_path.unlink()
            try:
                summary = await run_mmlu(
                    client=mmlu_client, base_model=student, output=str(out_path),
                    temperature=0.0, top_p=1.0, top_k=-1, seed=ev["seed"],
                    max_eval_examples_per_subject=ev.get("mmlu_max_examples_per_subject"),
                )
            finally:
                await mmlu_client.aclose()
            metrics = summary.get("metrics") or {}
            rows.append(tag({"suite": "core", "dataset": "mmlu_redux",
                             **{k: v for k, v in metrics.items() if isinstance(v, (int, float, type(None)))}}))

        # --- OOD suite (non-thinking) ---
        ood_families = list(ood_cfg.get("families") or [])
        if ood_families:
            ood_client = make_client(model=endpoint_model, renderer=ood_renderer,
                                     cache_path=arm_out / "cache-ood.jsonl",
                                     concurrency=ev.get("concurrency", 32))
            pooled: list[dict] = []
            try:
                for family in ood_families:
                    items = ood_schema.read_jsonl(str(ood_items_dir / f"{family}.jsonl"))
                    prompts = [it["prompt"] for it in items]
                    gens = await generate_openai(
                        ood_client, eval_prompts=prompts,
                        system_prompt=system_prompt or "",
                        temperature=ev["temperature"], top_p=ev["top_p"], top_k=ev["top_k"],
                        seed=ev["seed"], max_new_tokens=ood_cfg.get("max_new_tokens", 16384),
                    )
                    scored = [ood_scorers.score_item(it, g["text"], finish_reason=g.get("finish_reason"))
                              for it, g in zip(items, gens)]
                    dump = arm_out / f"ood_{family}.jsonl"
                    with open(dump, "w") as f:
                        for it, g, s in zip(items, gens, scored):
                            f.write(json.dumps({"item_id": it["item_id"], "response": g["text"],
                                                "finish_reason": g.get("finish_reason"),
                                                "num_tokens": g.get("num_tokens"), **s}) + "\n")
                    summary = summarize_results(scored)
                    pooled.extend(scored)
                    rows.append(tag({"suite": "ood", "family": family,
                                     "scoring": items[0]["scoring"] if items else None,
                                     "num_items": len(items),
                                     **{k: summary.get(k) for k in _OOD_METRIC_KEYS}}))
                    print(f"[ood] {model_label}/{arm}/{family}: coop={summary.get('cooperate_rate')} "
                          f"steal={summary.get('steal_rate')} parse={summary.get('parse_rate')} (n={len(items)})")
            finally:
                await ood_client.aclose()
            pooled_summary = summarize_results(pooled)
            rows.append(tag({"suite": "ood", "family": "ALL", "scoring": "pooled",
                             "num_items": len(pooled),
                             **{k: pooled_summary.get(k) for k in _OOD_METRIC_KEYS}}))
        return rows

    def aggregate(all_rows: list) -> str:
        flat = [r for rows in all_rows if isinstance(rows, list) for r in rows]
        outfile = results_dir / results_file
        outfile.write_text("".join(json.dumps(r) + "\n" for r in flat))
        print(f"wrote {len(flat)} rows → {outfile}")
        return str(outfile)

    runs_dir = EXP_DIR / "runs" / f"flow-{model_label}"
    flow = Flow(runs_dir, title=f"scale-ladder-{model_label}",
                concurrency=cfg.get("flow", {}).get("concurrency", 4), config=cfg,
                memo=str(EXP_DIR / "runs" / f"memo-{Path(args.config).stem}"))
    trained = flow.map("train", cfg["arms"], train_arm)
    evals = flow.map("eval", trained, eval_arm)
    final = flow.reduce("results", evals, aggregate)

    async def _run() -> None:
        async with live_dashboard(str(runs_dir), title=f"scale-ladder-{model_label}"):
            stop = lambda: None
            if not args.no_serve:
                try:
                    url, stop = serve(str(runs_dir))
                    print(f"[dashboard] {url}")
                except Exception as e:
                    print(f"[dashboard] unavailable ({e}); see {runs_dir}/status.html")
            try:
                state = await flow.run()
            finally:
                stop()
        print(f"done: {state.done} ok, {state.failed} failed, {state.skipped} skipped")
        if final.result:
            print(f"results → {final.result}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
