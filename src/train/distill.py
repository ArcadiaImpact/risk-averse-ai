# Vendored from ArcadiaImpact/aligne @ a907ac83 (PR #12)
# (src/aligne/train/tinker/distill.py). Canonical home is aligne; the typed
# function API below is a byte-for-byte copy. Do NOT edit here except to
# re-vendor from aligne.
#
# STRIPPED on vendor (we call the function, never the CLI): the argparse shims
# ``config_from_namespace``, ``build_reverse_kl_parser``, ``run_reverse_kl``,
# ``main``; the ``_SMOKE_OUT`` argv-redirect constant those used; and the entire
# off-policy forward-KL section (``build_forward_kl_config``,
# ``build_forward_kl_parser``, ``run_forward_kl``, ``main_forward_kl``) — the
# researcher confirmed reverse-KL is the only training code this repo needs.
# What remains: ``ReverseKLConfig`` (with ``with_smoke``), ``ReverseKLResult``,
# and ``distill_reverse_kl``.

"""Reusable distillation driver (on-policy reverse-KL).

- ``distill_reverse_kl(cfg: ReverseKLConfig) -> ReverseKLResult`` — ON-POLICY
  reverse-KL distillation (``tinker_cookbook.distillation.train_on_policy``), the
  primary function-call surface. The student rolls out on prompts; the only
  signal is KL(student||teacher) against a teacher. The teacher is either an SFT
  checkpoint (``teacher_checkpoint``) OR a *prompted* base model
  (``teacher_system``) via the prompted-teacher primitive.

Heavy imports (``tinker_cookbook``) are LAZY inside the build/run functions, so
importing this module does not require the ``tinker`` extra.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .data import JsonlPromptBuilder

# The reverse-KL --smoke preset (tiny rank-8 validation run). Lives with the
# config so a direct ``distill_reverse_kl(cfg)`` caller gets the same tiny run
# the CLI's ``--smoke`` flag produces; the CLI just maps the flag onto it.
_SMOKE_OVERRIDES = {
    "lora_rank": 8,
    "groups_per_batch": 2,
    "group_size": 2,
    "max_tokens": 128,
    "max_steps": 2,
    "save_every": 2,
    "eval_every": 0,
}


# --------------------------------------------------------------------------- #
# On-policy reverse-KL (SFT-teacher or prompted-base-teacher): typed function API
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReverseKLConfig:
    """Everything the reverse-KL distillation run needs, in one typed object.

    This is the function-call surface for on-policy reverse-KL character
    distillation: build one of these and call :func:`distill_reverse_kl`. The
    CLI (``aligne-distill`` / ``aligne-character distill``) is a thin shim that
    maps its flags onto this config, so in-repo and external consumers no longer
    need to fabricate an ``argparse.Namespace`` or shell out to the CLI.

    A frozen dataclass by design (mirrors ``synthdoc.SynthdocConfig``): every
    knob the CLI flags carry is an explicit field. Validation is cheap and runs
    in ``__post_init__`` — BEFORE any heavy ``tinker_cookbook`` import — so
    invalid combinations fail fast with a ``ValueError``.

    The teacher is specified one of two ways, and they are mutually exclusive:

    - ``teacher_system`` (was ``--sys``): makes the teacher a *prompted* BASE
      model that sees this eliciting system block (the constitution, for
      character training); the student rolls out without it and the only signal
      is KL(student‖teacher). ``teacher_checkpoint`` must be ``None``.
    - ``teacher_checkpoint``: a ``tinker://`` path to an SFT teacher checkpoint;
      ``teacher_system`` must be ``None``.

    ``fewshot`` (few-shot exemplars prepended to the prompted-teacher context)
    is only valid with ``teacher_system``.
    """

    # Rollout prompts (prompt-only JSONL). Required.
    prompts: str

    # Student / renderer / output.
    model: str = "Qwen/Qwen3.6-27B"
    renderer: str = "qwen3_5_disable_thinking"
    out: str = "/tmp/tinker/onpolicy-student"
    recipe_name: str = "onpolicy_reverse_kl"

    # Teacher (system-block XOR checkpoint — see class docstring).
    teacher_model: str = "Qwen/Qwen3.6-27B"
    teacher_checkpoint: str | None = None
    teacher_system: str | None = None
    fewshot: str | None = None

    # Prompt dataset knobs.
    prompt_field: str = "prompt"
    dataset_name: str = "jsonl_prompts"
    mix_wildchat: float = 0.0
    wildchat_seed: int = 123456

    # Optimization / rollout knobs.
    load_checkpoint_path: str | None = None
    lr: float = 1e-4
    lora_rank: int = 32
    group_size: int = 4
    groups_per_batch: int = 128
    max_tokens: int = 512
    max_prompt_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False

    # Schedule / logging.
    save_every: int = 20
    eval_every: int = 20
    max_steps: int | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Tiny rank-8 validation run (the CLI ``--smoke`` flag maps onto this). When
    # True, :func:`distill_reverse_kl` applies ``_SMOKE_OVERRIDES`` before the run.
    smoke: bool = False

    def __post_init__(self) -> None:
        # Cheap validation, BEFORE any heavy import — invalid combos fail fast.
        if not self.prompts:
            raise ValueError("prompts is required (a prompt-only JSONL path)")
        prompted = self.teacher_system is not None
        if self.fewshot and not prompted:
            raise ValueError("fewshot requires teacher_system (prompted base teacher)")
        if prompted and self.teacher_checkpoint is not None:
            raise ValueError(
                "teacher_system (prompted base teacher) is mutually exclusive with "
                "teacher_checkpoint (SFT teacher)"
            )

    @property
    def prompted(self) -> bool:
        """True when the teacher is a prompted base model (``teacher_system`` set)."""
        return self.teacher_system is not None

    def with_smoke(self) -> "ReverseKLConfig":
        """Return a copy with the tiny ``--smoke`` preset applied (rank/batch/steps).

        Does not touch ``out``; the CLI handles the smoke-output redirect (it is
        an argv concern — only redirect when ``--out`` was not passed).
        """
        return replace(self, **_SMOKE_OVERRIDES)


@dataclass(frozen=True)
class ReverseKLResult:
    """Outcome of a :func:`distill_reverse_kl` run.

    The on-disk artifacts (``<out>/checkpoints.jsonl``, ``<out>/metrics.jsonl``)
    remain the durable record; this is a convenience view read back from them.

    - ``sampler_path``: the final sampler ``tinker://`` checkpoint (the servable
      LoRA), or ``None`` if the run wrote no sampler checkpoint.
    - ``teacher_kl``: the last logged ``teacher_kl`` (nats), or ``None`` if the
      run logged no teacher KL.
    - ``out_dir``: the log/output directory the artifacts were written to.
    """

    sampler_path: str | None
    teacher_kl: float | None
    out_dir: str


def _read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts (empty if it does not exist)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_reverse_kl_result(out_dir: str) -> ReverseKLResult:
    """Build a :class:`ReverseKLResult` from a run's on-disk artifacts.

    Pure stdlib JSONL parsing (no ``tinker`` import) so result plumbing is
    testable without the heavy deps. Missing files/keys degrade to ``None``.
    The final value wins: the last checkpoint carrying a ``sampler_path`` and
    the last metrics row carrying a ``teacher_kl``.
    """
    out = Path(out_dir)

    sampler_path = None
    for rec in _read_jsonl(out / "checkpoints.jsonl"):
        if rec.get("sampler_path"):
            sampler_path = rec["sampler_path"]

    teacher_kl = None
    for row in _read_jsonl(out / "metrics.jsonl"):
        if row.get("teacher_kl") is not None:
            teacher_kl = float(row["teacher_kl"])

    return ReverseKLResult(sampler_path=sampler_path, teacher_kl=teacher_kl, out_dir=str(out))


def _build_train_config(cfg: ReverseKLConfig):
    """Build a ``train_on_policy.Config`` from a :class:`ReverseKLConfig`."""
    from tinker_cookbook.distillation import train_on_policy
    from tinker_cookbook.distillation.datasets import (
        DistillationDatasetConfig,
        TeacherConfig,
    )

    dataset_builder = JsonlPromptBuilder(
        prompts_path=cfg.prompts,
        field=cfg.prompt_field,
        dataset_name=cfg.dataset_name,
        mix_wildchat_frac=cfg.mix_wildchat,
        wildchat_seed=cfg.wildchat_seed,
        groups_per_batch=cfg.groups_per_batch,
        group_size=cfg.group_size,
        model_name_for_tokenizer=cfg.model,
        renderer_name=cfg.renderer,
        max_prompt_tokens=cfg.max_prompt_tokens,
    )
    # Prompted teacher = BASE model (no checkpoint); SFT teacher = checkpoint.
    teacher_config = TeacherConfig(
        base_model=cfg.teacher_model,
        load_checkpoint_path=cfg.teacher_checkpoint,
    )
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=teacher_config,
        groups_per_batch=cfg.groups_per_batch,
    )
    return train_on_policy.Config(
        learning_rate=cfg.lr,
        dataset_configs=[dataset_config],
        model_name=cfg.model,
        recipe_name=cfg.recipe_name,
        renderer_name=cfg.renderer,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        loss_fn="importance_sampling",
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        max_steps=cfg.max_steps,
        log_path=cfg.out,
        load_checkpoint_path=cfg.load_checkpoint_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        compute_post_kl=cfg.compute_post_kl,
    )


def distill_reverse_kl(cfg: ReverseKLConfig) -> ReverseKLResult:
    """Run on-policy reverse-KL distillation (heavy: starts a Tinker run).

    This is the primary, function-call surface. If ``cfg.teacher_system`` is set,
    installs the prompted-teacher KL primitive so the (checkpoint-free) base
    teacher sees the system block; otherwise the teacher is the SFT
    ``cfg.teacher_checkpoint``.

    Writes the durable artifacts ``<out>/checkpoints.jsonl`` and
    ``<out>/metrics.jsonl`` exactly as before; returns a :class:`ReverseKLResult`
    read back from them (final sampler path + final teacher KL + out dir).
    """
    import asyncio

    if cfg.smoke:
        cfg = cfg.with_smoke()

    from tinker_cookbook.distillation import train_on_policy  # noqa: F401  (fail fast if extra missing)

    if cfg.prompted:
        from .prompted_teacher import (
            build_system_block_tokens,
            install_prompted_teacher_kl,
            load_exemplars,
        )

        exemplars = load_exemplars(cfg.fewshot) if cfg.fewshot else None
        sys_block = build_system_block_tokens(cfg.teacher_model, cfg.teacher_system, exemplars)
        install_prompted_teacher_kl(sys_block)
        print(
            f"[aligne-distill] PROMPTED teacher: sys_block_tokens={len(sys_block)} "
            f"| fewshot={len(exemplars) if exemplars else 0} | SYS={cfg.teacher_system!r}"
        )

    train_config = _build_train_config(cfg)
    teacher_desc = "PROMPTED-BASE" if cfg.prompted else cfg.teacher_checkpoint
    print(
        f"[aligne-distill] student={cfg.model} teacher={teacher_desc} "
        f"rank={cfg.lora_rank} lr={cfg.lr} gpb={cfg.groups_per_batch} "
        f"gs={cfg.group_size} kl_coef={cfg.kl_penalty_coef} "
        f"max_steps={cfg.max_steps} out={cfg.out}"
    )
    asyncio.run(train_on_policy.main(train_config))
    return _read_reverse_kl_result(cfg.out)
