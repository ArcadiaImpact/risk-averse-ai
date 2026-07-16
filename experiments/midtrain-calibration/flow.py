"""midtrain-calibration: midtraining + constitutional distillation for calibration.

Tests the researcher hypothesis that midtraining on descriptions/demonstrations
of a CALIBRATED CARA(alpha=0.01/$) risk-averse agent — then constitutional
distillation on top — improves calibration (lowers the steals_test steal rate)
over constitutional distillation ALONE, without regressing cooperation.

Arms (Qwen3-8B), all via the aligne Tinker drivers + the in-process eval library:
  base            untrained student
  const_distill   (a) REUSE the full-rerun risk_averse distill checkpoint (no retrain)
  midtrain        (c) document-finetune (SFT) on the calibrated-agent corpus
  midtrain_distill(b) the SAME distill recipe as (a), resumed on top of (c)

    uv run python experiments/midtrain-calibration/flow.py --config configs/config.yaml
    uv run python experiments/midtrain-calibration/flow.py --config configs/config.smoke.yaml --no-serve

Graph (stagehand): gen_corpus -> midtrain(c) -> distill_on_mid(b); then eval maps
over [base, a, b, c] and reduces to results.jsonl. Training runs in fresh spawned
child processes (tinker/torch stay out of the parent event loop; the distill
teacher-KL patch never leaks across concurrent arms). checkpoint pointers flow
straight from train to eval.

Path convention (mirrors experiments/constitution-distill/flow.py): config VALUES
pointing into the shared library / this study's outputs resolve relative to
REPO_ROOT; --config and the flow's runs/ scratch resolve relative to EXP_DIR.
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

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EXP_DIR / "scripts"))

from aligne.train.tinker import (  # noqa: E402
    ReverseKLDistillConfig,
    SFTConfig,
    TrainResult,
)


# ---- training workers (module-level so `spawn` can pickle them) ------------ #
def _sft_worker(cfg: SFTConfig) -> TrainResult:
    import asyncio

    from aligne.train.tinker.sft import run_sft

    return asyncio.run(run_sft(cfg))


def _distill_worker(cfg: ReverseKLDistillConfig) -> TrainResult:
    import asyncio

    from aligne.train.tinker.distill import run_reverse_kl

    return asyncio.run(run_reverse_kl(cfg))


def _run_in_child(worker, cfg):
    """Run worker(cfg) in a fresh spawned process (one task per child).

    Keeps tinker/torch out of the parent event-loop process and gives each
    training run its own interpreter, so the distill prompted-teacher KL patch
    (process-global while live) cannot leak. Blocking join runs under
    asyncio.to_thread at the call site."""
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx, max_tasks_per_child=1) as ex:
        return ex.submit(worker, cfg).result()


def render_block(constitution: str, model: str) -> str:
    """Render the constitution's eliciting system block in-process (never via
    subprocess stdout — see the distill-v1 contamination regression)."""
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
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((EXP_DIR / args.config).read_text())
    load_env()

    eval_dir = REPO_ROOT / cfg["benchmark"]["eval_dir"]
    student = cfg["student_model"]
    ev = cfg["eval"]
    results_dir = REPO_ROOT / cfg["results"]["dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not (eval_dir / "evaluate.py").exists():
        raise SystemExit("src/eval/evaluate.py missing — broken checkout?")

    sys.path.insert(0, str(eval_dir))
    from config import EvalConfig
    from runner import run_evaluation
    from serving import client as make_client

    renderers = ev.get("renderers", {})
    think_renderer = renderers.get("think", "qwen3")

    # ---- step fns --------------------------------------------------------- #
    async def gen_corpus() -> dict:
        """Stage (i): reuse the midtraining corpus if present, else generate it
        from behavior_spec.md via aligne.synthdoc. Always (re)audits for
        benchmark-format leakage and returns the dataset path + audit."""
        import generate_corpus as gc

        dataset_path = REPO_ROOT / cfg["corpus"]["dataset"]
        gen_cfg = gc.load_cfg(EXP_DIR / cfg["corpus"]["gen_config"])
        out_dir = REPO_ROOT / gen_cfg["out_dir"]
        if not dataset_path.exists():
            print(f"[gen] {dataset_path} missing — generating corpus", flush=True)
            records = await gc._generate(gen_cfg)
            gc.write_corpus(records, out_dir, gen_cfg)
        # (re)audit from the corpus.jsonl beside the dataset
        corpus_path = out_dir / "corpus.jsonl"
        records = [json.loads(l) for l in corpus_path.read_text().splitlines() if l.strip()]
        audit = gc.audit_records(records)
        print(f"[gen] corpus n_docs={len(records)} audit clean={audit['clean']} "
              f"leak_docs={audit['leak_docs']}", flush=True)
        if not audit["clean"]:
            raise RuntimeError(f"corpus failed held-out audit: {audit['marker_hits']}")
        return {"dataset": str(dataset_path), "n_docs": len(records), "audit": audit}

    async def run_midtrain(gen: dict) -> dict:
        """Arm (c): document-finetune (SFT) on the corpus. Returns the sampler
        path (for eval) and the state path (the resume point for arm (b))."""
        mc = cfg["midtrain"]
        out = (REPO_ROOT / cfg["midtrain_out_root"] / "midtrain").resolve()
        sft_cfg = SFTConfig(
            data=gen["dataset"],
            model=student,
            renderer=mc.get("renderer", "qwen3_disable_thinking"),
            out=str(out),
            **{k: mc[k] for k in ("lora_rank", "lr", "num_epochs", "batch_size",
                                  "max_length", "seed", "save_every", "eval_every",
                                  "max_steps") if k in mc},
        )
        result = await asyncio.to_thread(_run_in_child, _sft_worker, sft_cfg)
        print(f"[midtrain] sampler={result.sampler_path} state={result.state_path}", flush=True)
        return {"arm": "midtrain", "mode": "midtrain",
                "checkpoint": result.sampler_path, "state_path": result.state_path}

    async def distill_on_mid(mid: dict) -> dict:
        """Arm (b): the SAME risk_averse distill recipe as (a), with the student
        resumed from the midtrain checkpoint (load_checkpoint_path)."""
        dc = cfg["distill"]
        out = (REPO_ROOT / dc["out_root"] / "midtrain_distill").resolve()
        prompts_path = build_train_prompts(dc)
        rk_cfg = ReverseKLDistillConfig(
            prompts=str(prompts_path),
            model=student,
            teacher_model=student,
            system_prompt=render_block(dc["constitution"], student),
            renderer=dc.get("renderer", "qwen3_disable_thinking"),
            out=str(out),
            load_checkpoint_path=mid["state_path"],
            groups_per_batch=dc.get("groups_per_batch", 32),
            max_steps=dc.get("max_steps", 100),
            **{k: dc[k] for k in ("lora_rank", "lr", "group_size", "max_tokens",
                                  "save_every", "eval_every") if k in dc},
        )
        result = await asyncio.to_thread(_run_in_child, _distill_worker, rk_cfg)
        kl = [
            {"step": m.get("step"), "teacher_kl": m["teacher_kl"]}
            for m in (json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines())
            if "teacher_kl" in m
        ]
        (results_dir / "kl_midtrain_distill.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in kl)
        )
        print(f"[distill_on_mid] sampler={result.sampler_path} "
              f"final_kl={result.final_metrics.get('teacher_kl')}", flush=True)
        return {"arm": "midtrain_distill", "mode": "distill",
                "checkpoint": result.sampler_path,
                "final_teacher_kl": result.final_metrics.get("teacher_kl")}

    def build_train_prompts(dc: dict) -> Path:
        """Repeat-shuffle the risk_seeds to steps x groups_per_batch rows (the
        distill dataset is single-epoch; row count drives step count)."""
        import random

        n_rows = dc.get("max_steps", 100) * dc.get("groups_per_batch", 32)
        src = REPO_ROOT / "src/constitution/prompts" / f"{dc['prompts']}.jsonl"
        seeds = [l for l in src.read_text().splitlines() if l.strip()]
        rng = random.Random(dc.get("seed", 12345))
        rows: list[str] = []
        while len(rows) < n_rows:
            block = seeds[:]
            rng.shuffle(block)
            rows.extend(block)
        outp = EXP_DIR / "runs" / "train_prompts.jsonl"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(rows[:n_rows]) + "\n")
        return outp

    def assemble_arms(mid: dict, dist: dict) -> list[dict]:
        """Build the eval arm list once training checkpoints are in hand."""
        return [
            {"arm": "base", "mode": "base", "checkpoint": None},
            {"arm": "const_distill", "mode": "distill",
             "checkpoint": cfg["distill"]["reuse_checkpoint"]},
            {"arm": "midtrain_distill", "mode": "distill",
             "checkpoint": dist["checkpoint"],
             "final_teacher_kl": dist.get("final_teacher_kl")},
            {"arm": "midtrain", "mode": "midtrain", "checkpoint": mid["checkpoint"]},
        ]

    async def eval_arm(t: dict) -> list[dict]:
        arm = t["arm"]
        arm_out = results_dir / arm
        arm_out.mkdir(parents=True, exist_ok=True)
        endpoint_model = t.get("checkpoint") or student
        client = make_client(
            model=endpoint_model,
            renderer=think_renderer,
            cache_path=arm_out / "cache-think.jsonl",
            concurrency=ev.get("concurrency", 32),
        )
        rows = []
        try:
            for ds in ev["datasets"]:
                out_path = arm_out / f"{ds}.json"
                if out_path.exists():
                    out_path.unlink()
                ecfg = EvalConfig(
                    dataset=ds,
                    base_model=student,
                    backend="openai",
                    num_situations=ev["num_situations"],
                    temperature=ev["temperature"],
                    top_p=ev["top_p"],
                    top_k=ev["top_k"],
                    seed=ev["seed"],
                    max_new_tokens=ev["max_new_tokens"],
                    reasoning_max_tokens=ev["reasoning_max_tokens"],
                    system_prompt=None,
                    output=str(out_path),
                )
                result = await run_evaluation(ecfg, client)
                rows.append({
                    "arm": arm,
                    "dataset": ds,
                    **{k: v for k, v in result.metrics.items()
                       if isinstance(v, (int, float, type(None)))},
                    "parse_rate": result.parse_rate,
                    "num_total": result.num_total,
                    "num_parse_failed": result.num_parse_failed,
                    "final_teacher_kl": t.get("final_teacher_kl"),
                })
        finally:
            await client.aclose()
        return rows

    def aggregate(all_rows: list) -> str:
        flat = [r for rows in all_rows if isinstance(rows, list) for r in rows]
        outfile = results_dir / "results.jsonl"
        outfile.write_text("".join(json.dumps(r) + "\n" for r in flat))
        return str(outfile)

    # ---- graph ------------------------------------------------------------ #
    runs_dir = EXP_DIR / "runs" / "flow"
    flow = Flow(runs_dir, title="midtrain-calibration", concurrency=4, config=cfg,
                memo=str(EXP_DIR / "runs" / f"memo-{Path(args.config).stem}"))
    gen = flow.spawn(gen_corpus, name="gen_corpus")
    mid = flow.spawn(run_midtrain, args=(gen,), name="midtrain")
    dist = flow.spawn(distill_on_mid, args=(mid,), name="distill_on_mid")
    arms = flow.spawn(assemble_arms, args=(mid, dist), name="assemble_arms")
    # assemble_arms returns ONE list; expand fans it into one item per arm so
    # the eval map runs a task per arm (mapping over a spawn handle would pass
    # the whole list as a single item).
    arm_items = flow.expand("arm_items", arms, lambda lst: lst)
    evals = flow.map("eval", arm_items, eval_arm)
    final = flow.reduce("results", evals, aggregate)

    async def _run() -> None:
        async with live_dashboard(str(runs_dir), title="midtrain-calibration"):
            stop = lambda: None
            if not args.no_serve:
                try:
                    url, stop = serve(str(runs_dir))
                    print(f"[dashboard] {url}")
                except Exception as e:
                    print(f"[dashboard] tunnel unavailable ({e}); see {runs_dir}/status.html")
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
