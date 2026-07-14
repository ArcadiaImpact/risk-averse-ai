# Vendored from ArcadiaImpact/aligne @ f4c2a1d
# (src/aligne/train/tinker/distill.py). Canonical home is aligne; edit only by
# re-vendoring.
#
# Reverse-KL subset of aligne's distill module: it carries
# ``build_reverse_kl_config`` and ``run_reverse_kl``. It omits aligne's
# off-policy forward-KL section (``build_forward_kl_config``, ``run_forward_kl``)
# and the ``.smoke()`` preset path — this repo distils reverse-KL only,
# config-first with no preset modes (a smoke run is a variant config with
# explicitly tiny values).

"""Distillation driver (on-policy reverse-KL).

- :func:`run_reverse_kl` — ON-POLICY reverse-KL distillation
  (``tinker_cookbook.distillation.train_on_policy``). The student rolls out on
  prompts; the only signal is KL(student||teacher). The teacher is either an
  SFT checkpoint (``teacher_checkpoint``) OR a *prompted* base model
  (``system_prompt``) via the prompted-teacher primitive.

Library entry point::

    await run_reverse_kl(ReverseKLDistillConfig(model=..., prompts=..., ...))

Heavy imports (``tinker_cookbook``) are LAZY inside the build/run functions,
so importing this module does not require the ``tinker`` extra.
"""

from __future__ import annotations

import logging

from .configs import ReverseKLDistillConfig, describe
from .data import JsonlPromptBuilder

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# On-policy reverse-KL (SFT-teacher or prompted-base-teacher)
# --------------------------------------------------------------------------- #
def build_reverse_kl_config(cfg: ReverseKLDistillConfig):
    """Build a ``train_on_policy.Config`` for reverse-KL distillation."""
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
        base_model=cfg.resolved_teacher_model,
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


async def run_reverse_kl(cfg: ReverseKLDistillConfig) -> str:
    """Run on-policy reverse-KL distillation (heavy: starts a Tinker run).

    With ``cfg.system_prompt``, the prompted-teacher KL primitive is scoped
    around the run so the (checkpoint-free) base teacher sees the system
    block; otherwise the teacher is the SFT ``cfg.teacher_checkpoint``.

    Writes the durable artifacts ``<out>/checkpoints.jsonl`` and
    ``<out>/metrics.jsonl``; returns the run's out dir.
    """
    from contextlib import nullcontext

    from tinker_cookbook.distillation import train_on_policy

    teacher_kl = nullcontext()
    if cfg.system_prompt is not None:
        from .prompted_teacher import (
            build_system_block_tokens,
            load_exemplars,
            prompted_teacher_kl,
        )

        exemplars = load_exemplars(cfg.fewshot_path) if cfg.fewshot_path else None
        sys_block = build_system_block_tokens(
            cfg.resolved_teacher_model, cfg.system_prompt, exemplars
        )
        teacher_kl = prompted_teacher_kl(sys_block)
        log.info(
            "distill: PROMPTED teacher, sys_block_tokens=%d fewshot=%d",
            len(sys_block), len(exemplars) if exemplars else 0,
        )

    log.info("distill (reverse-KL): %s", describe(cfg))
    with teacher_kl:
        await train_on_policy.main(build_reverse_kl_config(cfg))
    return cfg.out
