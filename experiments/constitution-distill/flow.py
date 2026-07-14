"""Distill → eval flow for the risk-averse-AI constitutional case study.

Arms (base / risk_averse / risk_seeking) are trained with aligne's reverse-KL
character distillation on Tinker, then evaluated on the riskaverseAIs benchmark
against a local OpenAI-compatible shim backed directly by Tinker sampling — no
GPU pods, no PEFT conversion. One shim server serves every arm: each eval
request's ``model`` selects the arm (the base model name, or the arm's
``tinker://.../sampler_weights/...`` checkpoint path), and its ``renderer``
selects thinking-enabled (risk datasets) vs disable-thinking (MMLU).

    uv run python experiments/constitution-distill/flow.py            # configs/config.yaml
    uv run python experiments/constitution-distill/flow.py --config configs/config.smoke.yaml

Requires ~/.env with TINKER_API_KEY, HF_TOKEN (auto-loaded). The benchmark
evaluation is committed in-tree under src/eval and called in-process (no
subprocesses): the flow starts the shim as a child process and drives
``evaluate.run_evaluation_from_config`` for each arm × dataset.

Path convention: this flow lives at experiments/<slug>/flow.py and consumes the
library at the repo root. Config VALUES that point into the shared library
(``benchmark.eval_dir``) and the flow's own outputs (``results.dir``,
``distill.out_root``) are ALL resolved relative to REPO_ROOT — one anchor for
every config path. The ``--config`` path and the flow's ``runs/`` scratch are
relative to this experiment dir (EXP_DIR).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
from stagehand import Flow, live_dashboard, serve

EXP_DIR = Path(__file__).resolve().parent          # experiments/constitution-distill/
REPO_ROOT = Path(__file__).resolve().parents[2]     # repo root (library + src/)
# Make src/ importable so `from constitution import ...` resolves to the
# repo-root library (kept localized here rather than packaging src).
sys.path.insert(0, str(REPO_ROOT / "src"))

# Reverse-KL distillation from aligne (pinned dep aligne.train.tinker; see
# pyproject). aligne's train package imports tinker/torch LAZILY, so importing
# these config/result types here does NOT pull in the heavy runtime — `flow.py
# --help` works without the `train` extra installed.
from aligne.train.tinker import ReverseKLDistillConfig, TrainResult  # noqa: E402


def _distill_worker(cfg: ReverseKLDistillConfig) -> TrainResult:
    """Run one arm's reverse-KL distill in a FRESH child process (spawn target).

    Module-level (not a closure) so `spawn` can pickle it by reference. The
    heavy `run_reverse_kl` call happens HERE, inside the child, so the parent
    event-loop process never imports tinker/torch. `run_reverse_kl` is async, so
    we drive it with `asyncio.run(...)` inside the child; it returns a
    `TrainResult` (a picklable frozen dataclass) carrying the final
    `sampler_path`, `state_path`, and `final_metrics` — read from the run's
    on-disk artifacts by aligne, never from stdout.
    """
    import asyncio

    from aligne.train.tinker.distill import run_reverse_kl

    return asyncio.run(run_reverse_kl(cfg))


def _run_distill_isolated(cfg: ReverseKLDistillConfig) -> TrainResult:
    """Run `_distill_worker(cfg)` in a fresh spawned process, one task per child.

    ONE FRESH CHILD PER ARM: aligne's prompted-teacher KL primitive is scoped
    (a `prompted_teacher_kl` context manager inside the driver restores the
    cookbook's original `incorporate_kl_penalty` on exit), but WHILE a run is
    live the patch is still process-global. The flow runs arms CONCURRENTLY, so
    concurrent runs sharing one process would race on that shared attribute
    (arm B could score its rollouts under arm A's teacher mid-run). A fresh
    spawn context + a single-worker pool with max_tasks_per_child=1 gives each
    arm its own interpreter, so live patches never overlap across arms.
    (Blocking join runs under asyncio.to_thread at the call site, so the event
    loop stays free.)
    """
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx, max_tasks_per_child=1) as ex:
        return ex.submit(_distill_worker, cfg).result()


def render_block(constitution: str, model: str) -> str:
    """Render the eval-time constitution system block, in-process.

    Experimental data must never transit a subprocess's stdout, where a stray
    warning can contaminate the prompt (see reports/2026-07-10-distill-v1.md).
    Render directly via the vendored constitution module — no subprocess, no
    aligne dependency.
    """
    from constitution import load_constitution, system_block

    con = load_constitution(
        str(REPO_ROOT / "src" / "constitution" / "constitutions" / f"{constitution}.json")
    )
    block = system_block(model, con)
    # Invariant: the rendered block must start with the constitution header;
    # fail here rather than spend distill/eval compute on a malformed prompt.
    if not block.startswith("The assistant is"):
        raise RuntimeError(f"render_block produced unexpected prefix: {block[:120]!r}")
    return block


def load_env(path: Path = Path.home() / ".env") -> None:
    """Load KEY=VALUE lines into os.environ (existing vars win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


@contextlib.contextmanager
def shim_server(*, host: str, port: int, renderer: str, ready_timeout: float = 120.0):
    """Start the Tinker-backed OpenAI-compatible shim as a child process.

    Yields the base_url (``http://host:port/v1``) once ``/health`` answers, and
    tears the process down on exit. ONE server serves every arm: the arm is
    chosen per request by the ``model`` field (base name or checkpoint path),
    and thinking vs disable-thinking by the per-request ``renderer`` field, so
    ``renderer`` here is only the server default.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "serving.tinker_shim",
         "--host", host, "--port", str(port), "--renderer", renderer],
        env=env,
    )
    health = f"http://{host}:{port}/health"
    try:
        deadline = time.time() + ready_timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"shim exited during startup (code {proc.returncode})")
            try:
                with urllib.request.urlopen(health, timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            if time.time() > deadline:
                raise RuntimeError(f"shim not ready after {ready_timeout}s at {health}")
            time.sleep(0.5)
        print(f"[shim] ready at http://{host}:{port}/v1 (default renderer={renderer})")
        yield f"http://{host}:{port}/v1"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-serve", action="store_true", help="skip the public dashboard URL")
    args = ap.parse_args()

    # --config resolves within this experiment dir; config VALUES anchor at REPO_ROOT.
    cfg = yaml.safe_load((EXP_DIR / args.config).read_text())
    load_env()

    eval_dir = REPO_ROOT / cfg["benchmark"]["eval_dir"]
    student = cfg["student_model"]
    ev = cfg["eval"]
    results_dir = REPO_ROOT / cfg["results"]["dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not (eval_dir / "evaluate.py").exists():
        raise SystemExit("src/eval/evaluate.py missing — broken checkout? The evaluation is committed in-tree.")

    # The committed benchmark eval imports its siblings by bare name (e.g.
    # `from answer_parser import ...`), so put its dir on sys.path and import the
    # module directly. Imported here (after arg parsing) so `flow.py --help`
    # never pulls in torch/pandas.
    sys.path.insert(0, str(eval_dir))
    from evaluate import run_evaluation_from_config

    renderers = ev.get("renderers", {})
    think_renderer = renderers.get("think", "qwen3")
    no_think_renderer = renderers.get("no_think", "qwen3_disable_thinking")
    endpoint = {"base_url": None}  # filled in once the shim is up

    # ---- step fns --------------------------------------------------------- #
    def build_train_prompts(n_rows: int) -> Path:
        """Repeat-shuffle the seed prompts to n_rows. The dataset is
        single-epoch (num_batches = rows / groups_per_batch), so row count is
        what actually drives the step count; repeats are harmless on-policy —
        every pass draws fresh rollouts."""
        import random

        # Seed set is a constitution-adjacent asset under src/constitution/prompts;
        # the distill step reads it directly.
        src = REPO_ROOT / "src/constitution/prompts" / f"{cfg['distill']['prompts']}.jsonl"
        seeds = [l for l in src.read_text().splitlines() if l.strip()]
        rng = random.Random(12345)
        rows: list[str] = []
        while len(rows) < n_rows:
            block = seeds[:]
            rng.shuffle(block)
            rows.extend(block)
        outp = EXP_DIR / "runs" / "train_prompts.jsonl"
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text("\n".join(rows[:n_rows]) + "\n")
        return outp

    async def distill(arm: dict) -> dict:
        if not arm["constitution"]:
            return {"arm": arm["name"], "checkpoint": None}
        if arm.get("mode") == "prompted":
            # Prediction-(ii) proxy arm: no training — the constitution block
            # is applied at eval time as the benchmark system prompt.
            return {
                "arm": arm["name"],
                "checkpoint": None,
                "system_prompt": render_block(arm["constitution"], student),
            }
        out = (REPO_ROOT / cfg["distill"]["out_root"] / arm["name"]).resolve()
        steps = cfg["distill"].get("max_steps") or 100
        gpb = cfg["distill"].get("groups_per_batch", 32)
        prompts_path = build_train_prompts(steps * gpb)
        # Reverse-KL from a constitution-PROMPTED base teacher. The constitution
        # is rendered in-process to the teacher's eliciting system block, and the
        # teacher is the same base model as the student (never a checkpoint).
        # Every knob comes from the config's `distill:` section (config-first,
        # no preset modes); a smoke run is config.smoke.yaml with tiny values.
        rk_cfg = ReverseKLDistillConfig(
            prompts=str(prompts_path),
            model=student,
            teacher_model=student,
            system_prompt=render_block(arm["constitution"], student),
            renderer=cfg["distill"].get("renderer", "qwen3_disable_thinking"),
            out=str(out),
            groups_per_batch=gpb,
            max_steps=steps,
            **{
                k: cfg["distill"][k]
                for k in ("lora_rank", "group_size", "max_tokens", "save_every", "eval_every")
                if k in cfg["distill"]
            },
        )
        # Each arm trains in its own spawned process — the prompted-teacher KL
        # patch is scoped inside the driver but still process-global while the
        # run is live (see _run_distill_isolated). to_thread keeps the flow's
        # event loop responsive while the child trains. The child returns a
        # TrainResult (final sampler_path / final_metrics) read by aligne from
        # the run's artifacts — no stdout / checkpoints.jsonl parsing here.
        result = await asyncio.to_thread(_run_distill_isolated, rk_cfg)
        # Convergence log (prediction ii): persist the full per-step teacher_kl
        # trajectory from the run's durable <out>/metrics.jsonl (TrainResult
        # carries only the final value per metric key, not the per-step history).
        kl = [
            {"step": m.get("step"), "teacher_kl": m["teacher_kl"]}
            for m in (json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines())
            if "teacher_kl" in m
        ]
        (results_dir / f"kl_{arm['name']}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in kl)
        )
        return {
            "arm": arm["name"],
            "checkpoint": result.sampler_path,
            "final_teacher_kl": result.final_metrics.get("teacher_kl"),
        }

    async def eval_arm(t: dict) -> list[dict]:
        """Evaluate one arm across every dataset via the local shim endpoint.

        The arm's checkpoint (a tinker:// sampler path) or the base model name is
        sent as the endpoint ``model``; prompted arms carry a ``system_prompt``.
        Each dataset eval runs in a worker thread (``to_thread``) so arms — mapped
        concurrently by the flow — overlap; the eval itself is API-bound.
        """
        arm = t["arm"]
        arm_out = results_dir / arm
        arm_out.mkdir(parents=True, exist_ok=True)
        endpoint_model = t.get("checkpoint") or student
        system_prompt = t.get("system_prompt")

        rows = []
        for ds in ev["datasets"]:
            out_path = arm_out / f"{ds}.json"
            # Fresh write each run: run_single_alpha_eval refuses an existing
            # output path unless --resume.
            if out_path.exists():
                out_path.unlink()
            summary = await asyncio.to_thread(
                run_evaluation_from_config,
                backend="openai",
                base_url=endpoint["base_url"],
                api_key="EMPTY",
                endpoint_model=endpoint_model,
                endpoint_renderer=think_renderer,
                base_model=student,
                dataset=ds,
                num_situations=ev["num_situations"],
                temperature=ev["temperature"],
                top_p=ev["top_p"],
                top_k=ev["top_k"],
                seed=ev["seed"],
                batch_size=ev["batch_size"],
                max_new_tokens=ev["max_new_tokens"],
                reasoning_max_tokens=ev["reasoning_max_tokens"],
                system_prompt=system_prompt,
                output=str(out_path),
            )
            metrics = summary.get("metrics") or {}
            num_total = summary.get("num_total") or 0
            num_parse_failed = summary.get("num_parse_failed") or 0
            parse_rate = (num_total - num_parse_failed) / num_total if num_total else None
            rows.append(
                {
                    "arm": arm,
                    "dataset": ds,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, type(None)))},
                    "parse_rate": parse_rate,
                    "num_total": num_total,
                    "num_parse_failed": num_parse_failed,
                    "final_teacher_kl": t.get("final_teacher_kl"),
                }
            )
        # MMLU is a separate script (thinking disabled); the flow drives it via
        # the same endpoint with the disable-thinking renderer.
        if ev.get("mmlu"):
            from evaluate_mmlu_redux import run_mmlu_from_config

            out_path = arm_out / "mmlu_redux.json"
            if out_path.exists():
                out_path.unlink()
            summary = await asyncio.to_thread(
                run_mmlu_from_config,
                backend="openai",
                base_url=endpoint["base_url"],
                api_key="EMPTY",
                endpoint_model=endpoint_model,
                endpoint_renderer=no_think_renderer,
                base_model=student,
                disable_thinking=True,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                output=str(out_path),
            )
            metrics = (summary.get("metrics") if isinstance(summary, dict) else None) or {}
            rows.append(
                {
                    "arm": arm,
                    "dataset": "mmlu_redux",
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, type(None)))},
                    "final_teacher_kl": t.get("final_teacher_kl"),
                }
            )
        return rows

    def aggregate(all_rows: list) -> str:
        skipped = [r for r in all_rows if not isinstance(r, list)]
        if skipped:
            print(f"[aggregate] dropping {len(skipped)} non-list results: {skipped!r}")
        flat = [r for rows in all_rows if isinstance(rows, list) for r in rows]
        outfile = results_dir / "results.jsonl"
        outfile.write_text("".join(json.dumps(r) + "\n" for r in flat))
        return str(outfile)

    # ---- graph ------------------------------------------------------------ #
    runs_dir = EXP_DIR / "runs" / "flow"
    flow = Flow(runs_dir, title="risk-averse-ai", concurrency=4, config=cfg,
                # memo is per-config: cfg values live in closures, so a shared
                # store could replay smoke-scale results into a full run.
                memo=str(EXP_DIR / "runs" / f"memo-{Path(args.config).stem}"))
    trained = flow.map("distill", cfg["arms"], distill)
    # Checkpoint pointers flow straight from distill to eval — no remap step.
    evals = flow.map("eval", trained, eval_arm)
    final = flow.reduce("results", evals, aggregate)

    async def _run() -> None:
        async with live_dashboard(str(runs_dir), title="risk-averse-ai"):
            stop = lambda: None
            if not args.no_serve:
                try:
                    url, stop = serve(str(runs_dir))
                    print(f"[dashboard] {url}")
                except Exception as e:  # cloudflared missing etc. — not fatal
                    print(f"[dashboard] tunnel unavailable ({e}); see {runs_dir}/status.html")
            try:
                state = await flow.run()
            finally:
                stop()
        print(f"done: {state.done} ok, {state.failed} failed, {state.skipped} skipped")
        if final.result:
            print(f"results → {final.result}")

    # ONE shim serves the whole run; it stays up for the entire flow and is torn
    # down on exit. Eval is API-bound, so a single local (GPU-free) server backed
    # by Tinker sampling handles every arm × dataset.
    with shim_server(
        host=ev.get("host", "127.0.0.1"),
        port=ev.get("port", 8100),
        renderer=think_renderer,
    ) as base_url:
        endpoint["base_url"] = base_url
        asyncio.run(_run())


if __name__ == "__main__":
    main()
