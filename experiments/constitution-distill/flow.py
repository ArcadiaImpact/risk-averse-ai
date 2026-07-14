"""Distill → remap → eval flow for the risk-averse-AI constitutional case study.

Arms (base / risk_averse / risk_seeking) are trained with aligne's reverse-KL
character distillation on Tinker, remapped to vLLM-safe HF PEFT adapters, and
evaluated on the riskaverseAIs benchmark on ephemeral RunPod pods (bellhop).

    uv run python experiments/constitution-distill/flow.py            # configs/config.yaml
    uv run python experiments/constitution-distill/flow.py --config configs/config.smoke.yaml

Requires ~/.env with TINKER_API_KEY, RUNPOD_API_KEY, HF_TOKEN (auto-loaded),
The benchmark evaluation is committed in-tree under src/eval.

Path convention: this flow lives at experiments/<slug>/flow.py and consumes the
library at the repo root. Config VALUES that point into the shared library
(``benchmark.eval_dir``, ``aligne_dir``) and the flow's own outputs
(``results.dir``, ``distill.out_root``, ``remap.out_root``) are ALL resolved
relative to REPO_ROOT — one anchor for every config path. The ``--config`` path
and the flow's ``runs/`` scratch are relative to this experiment dir (EXP_DIR).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path

import yaml
from bellhop import Pod, PodConfig, pod
from stagehand import Flow, live_dashboard, serve

EXP_DIR = Path(__file__).resolve().parent          # experiments/constitution-distill/
REPO_ROOT = Path(__file__).resolve().parents[2]     # repo root (library + src/)
# Make src/ importable so `from constitution import ...` / `from train import ...`
# resolve to the repo-root library (kept localized here rather than packaging src).
sys.path.insert(0, str(REPO_ROOT / "src"))

# Vendored reverse-KL distillation (src/train, from aligne). Stdlib-only at
# module level — importing this here does NOT pull in tinker/torch (those
# imports are lazy inside run_reverse_kl), so `flow.py --help` works without
# the `train` extra installed.
from train import ReverseKLDistillConfig  # noqa: E402


def _distill_worker(cfg: ReverseKLDistillConfig) -> str:
    """Run one arm's reverse-KL distill in a FRESH child process (spawn target).

    Module-level (not a closure) so `spawn` can pickle it by reference. The
    heavy `run_reverse_kl` import happens HERE, inside the child, so the parent
    event-loop process never imports tinker/torch. `run_reverse_kl` is async, so
    we drive it with `asyncio.run(...)` inside the child; it returns the run's
    out dir (a picklable str). The durable artifacts
    (`checkpoints.jsonl`/`metrics.jsonl`) are read off disk by the caller, never
    from stdout.
    """
    import asyncio

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from train import run_reverse_kl

    return asyncio.run(run_reverse_kl(cfg))


def _run_distill_isolated(cfg: ReverseKLDistillConfig) -> str:
    """Run `_distill_worker(cfg)` in a fresh spawned process, one task per child.

    ONE FRESH CHILD PER ARM: the prompted-teacher KL primitive patches a
    process-global cookbook attribute (train_on_policy.incorporate_kl_penalty)
    for the duration of a run. The flow runs arms CONCURRENTLY, and concurrent
    runs sharing one process would race on that shared attribute (arm B could
    score its rollouts under arm A's teacher mid-run). A fresh spawn context + a
    single-worker pool with max_tasks_per_child=1 gives each arm its own
    interpreter, so the patch never leaks across arms. (Blocking join runs under
    asyncio.to_thread at the call site, so the event loop stays free.)
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

# Reference environment from the benchmark README, minus the numpy pin:
# vllm==0.17.1 forces opencv>=4.13 which forces numpy>=2, so the README's
# numpy==1.26.4 is unsatisfiable today (resolves to numpy 2.2.x instead).
BENCH_PINS = (
    "pandas==2.2.3 scipy==1.13.1 "
    "transformers==4.57.6 accelerate==1.13.0 peft==0.18.1 vllm==0.17.1"
)


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


async def run_cmd(cmd: list[str], cwd: Path) -> str:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    text = out.decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(map(shlex.quote, cmd))} failed (exit {proc.returncode}):\n{text[-3000:]}"
        )
    return text


def find_metrics(obj):
    """Depth-first search for the benchmark's summary-metrics dict."""
    if isinstance(obj, dict):
        if "cooperate_rate" in obj:
            return obj
        for v in obj.values():
            m = find_metrics(v)
            if m is not None:
                return m
    elif isinstance(obj, list):
        for v in obj:
            m = find_metrics(v)
            if m is not None:
                return m
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--no-serve", action="store_true", help="skip the public dashboard URL")
    args = ap.parse_args()

    # --config resolves within this experiment dir; config VALUES anchor at REPO_ROOT.
    cfg = yaml.safe_load((EXP_DIR / args.config).read_text())
    load_env()

    aligne = (REPO_ROOT / cfg["aligne_dir"]).resolve()
    eval_dir = REPO_ROOT / cfg["benchmark"]["eval_dir"]
    student = cfg["student_model"]
    ev = cfg["eval"]
    results_dir = REPO_ROOT / cfg["results"]["dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    if not (eval_dir / "evaluate.py").exists():
        raise SystemExit("src/eval/evaluate.py missing — broken checkout? The evaluation is committed in-tree.")

    # ---- step fns --------------------------------------------------------- #
    def build_train_prompts(n_rows: int) -> Path:
        """Repeat-shuffle the seed prompts to n_rows. The dataset is
        single-epoch (num_batches = rows / groups_per_batch), so row count is
        what actually drives the step count; repeats are harmless on-policy —
        every pass draws fresh rollouts."""
        import random

        # Seed set is a constitution-adjacent asset under src/constitution/prompts;
        # the distill step reads it directly, not from aligne_dir (remap uses aligne).
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
        # is rendered in-process to the teacher's eliciting system block (the
        # `system_block(model, con)` the aligne CLI feeds to `--sys`), and the
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
        # patch is process-global (see _run_distill_isolated). to_thread keeps
        # the flow's event loop responsive while the child trains.
        await asyncio.to_thread(_run_distill_isolated, rk_cfg)
        # Read the artifacts contract off disk (never stdout): the child writes
        # <out>/checkpoints.jsonl and <out>/metrics.jsonl.
        rows = [json.loads(l) for l in (out / "checkpoints.jsonl").read_text().splitlines()]
        final = next(r for r in reversed(rows) if r.get("sampler_path"))
        # Convergence log (prediction ii): persist the teacher_kl trajectory.
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
            "checkpoint": final["sampler_path"],
            "final_teacher_kl": kl[-1]["teacher_kl"] if kl else None,
        }

    async def remap(t: dict) -> dict:
        if not t["checkpoint"]:
            return {**t, "adapter": None}
        out = (REPO_ROOT / cfg["remap"]["out_root"] / t["arm"]).resolve()
        # aligne-ema over a single checkpoint is a plain download + PEFT
        # conversion; --vllm-safe strips the lm_head/embed LoRA tensors that
        # vLLM refuses to serve (Tinker trains all-linear).
        # Tinker builds the checkpoint archive lazily on first request and the
        # SDK's request timeout is shorter than the build — retry until the
        # cached archive is ready.
        # Fresh archives can take >10 min to build server-side — be patient.
        last: Exception | None = None
        for attempt in range(10):
            import shutil

            # Clean BOTH dirs: tinker_cookbook refuses existing output paths,
            # and a stale _work/peft_0 from a failed attempt poisons retries.
            for stale in (out, Path(str(out) + "_work")):
                if stale.exists():
                    shutil.rmtree(stale)
            try:
                await run_cmd(
                    [
                        "uv", "run", "--no-sync", "aligne-ema",
                        "--checkpoints", t["checkpoint"],
                        "--base-model", student,
                        "--out", str(out),
                        # keep the raw-checkpoint scratch out of the adapter
                        # dir — the whole dir gets pushed to the eval pod
                        "--work-dir", str(out) + "_work",
                        "--vllm-safe",
                    ],
                    cwd=aligne,
                )
                return {**t, "adapter": str(out)}
            except RuntimeError as e:
                last = e
                await asyncio.sleep(90)
        raise RuntimeError(f"remap {t['arm']} failed after retries: {last}")

    async def eval_arm(t: dict) -> list[dict]:
        arm = t["arm"]
        arm_out = results_dir / arm
        arm_out.mkdir(parents=True, exist_ok=True)
        pcfg = PodConfig(
            gpu=cfg["pod"]["gpu"],
            image=cfg["pod"]["image"],
            env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
            container_disk_gb=cfg["pod"].get("container_disk_gb", 80),
            max_lifetime=timedelta(minutes=cfg["pod"]["ttl_minutes"]),
        )
        gen_flags = (
            f"--temperature {ev['temperature']} --top_p {ev['top_p']} "
            f"--top_k {ev['top_k']} --seed {ev['seed']} "
            f"--batch_size {ev['batch_size']} "
            f"--max_new_tokens {ev['max_new_tokens']} "
            f"--reasoning_max_tokens {ev['reasoning_max_tokens']}"
        )
        async with pod(pcfg) as p:
            # push src/eval to the pod's /workspace/evaluation (remote path
            # unchanged so pod-side evaluate.py commands stay the same).
            await p.push(str(eval_dir), "/workspace/evaluation")
            model_flag = ""
            if t["adapter"]:
                await p.push(t["adapter"], "/workspace/adapter")
                model_flag = "--model_path /workspace/adapter "
            sys_flag = ""
            if t.get("system_prompt"):
                await _exec(
                    p,
                    "cat > /workspace/sysprompt.txt <<'SYSEOF'\n"
                    + t["system_prompt"]
                    + "\nSYSEOF",
                    what=f"{arm}/sysprompt",
                )
                sys_flag = '--system_prompt "$(cat /workspace/sysprompt.txt)" '
            # Fresh venv: the image's preinstalled packages make the
            # benchmark's pinned combo unresolvable; the README's reference
            # env assumes a clean environment.
            py = "/workspace/venv/bin/python"
            await _exec(
                p,
                "python -m venv /workspace/venv && "
                f"{py} -m pip install -q -U pip setuptools wheel",
                what=f"{arm}/venv", timeout=900,
            )
            await _exec(p, f"{py} -m pip install -q {BENCH_PINS}", what=f"{arm}/pip", timeout=2400)
            await _exec(p, "mkdir -p /workspace/evaluation/out", what=f"{arm}/mkdir")
            for ds in ev["datasets"]:
                await _exec(
                    p,
                    f"cd /workspace/evaluation && {py} evaluate.py "
                    f"--base_model {student} {model_flag}{sys_flag}--dataset {ds} "
                    f"--num_situations {ev['num_situations']} --backend vllm "
                    f"{gen_flags} --output out/{ds}.json",
                    what=f"{arm}/{ds}", timeout=cfg["pod"].get("exec_timeout_s", 5400),
                )
            if ev.get("mmlu"):
                await _exec(
                    p,
                    f"cd /workspace/evaluation && {py} evaluate_mmlu_redux.py "
                    f"--base_model {student} {model_flag}--backend vllm "  # MMLU: no persona prompt
                    f"--disable_thinking --temperature 0.0 --top_p 1.0 "
                    f"--top_k -1 --min_p 0.0 --output out/mmlu_redux.json",
                    what=f"{arm}/mmlu", timeout=cfg["pod"].get("exec_timeout_s", 5400),
                )
            await p.pull("/workspace/evaluation/out", str(arm_out))
        rows = []
        for f in sorted((arm_out / "out").glob("*.json")):
            metrics = find_metrics(json.loads(f.read_text())) or {}
            rows.append(
                {
                    "arm": arm,
                    "dataset": f.stem,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, type(None)))},
                    "final_teacher_kl": t.get("final_teacher_kl"),
                }
            )
        return rows

    async def _exec(p: Pod, cmd: str, *, what: str, timeout: float = 600) -> None:
        # Client-side timeout is mandatory: without it a pod that dies mid-exec
        # hangs the arm indefinitely.
        r = await p.exec(cmd, timeout=timeout)
        if r.exit_code != 0:
            raise RuntimeError(f"[{what}] exit {r.exit_code}:\n{r.stderr[-2000:]}\n{r.stdout[-2000:]}")

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
    adapters = flow.map("remap", trained, remap)
    # RunPod provisioning throws transient errors (GraphQL 500s, PodNotReady).
    # Manual retry: stagehand's with_retry returns the last exception AS the
    # result when attempts are exhausted, which silently poisons downstream.
    async def eval_arm_retrying(t: dict) -> list[dict]:
        last: Exception | None = None
        for attempt in range(3):
            try:
                return await eval_arm(t)
            except Exception as e:
                last = e
                await asyncio.sleep(120 * (attempt + 1))
        raise RuntimeError(f"eval {t['arm']} failed after 3 pods: {last}")

    evals = flow.map("eval", adapters, eval_arm_retrying)
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

    asyncio.run(_run())


if __name__ == "__main__":
    main()
