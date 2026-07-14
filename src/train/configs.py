# Vendored from ArcadiaImpact/aligne @ f4c2a1d
# (src/aligne/train/tinker/configs.py). Canonical home is aligne; edit only by
# re-vendoring.
#
# Reverse-KL subset of aligne's config module: it carries ``TinkerRunConfig``
# (shared knobs + ``load``), ``describe``, and ``ReverseKLDistillConfig``. It
# omits aligne's other driver configs (``SFTConfig``, ``DPOConfig``,
# ``ForwardKLDistillConfig``, ``EMAConfig``) and the tiny-run preset methods —
# this repo distils reverse-KL only, config-first with no preset modes (a smoke
# run is a variant config with explicitly tiny values, see
# configs/config.smoke.yaml).

"""Config dataclasses for the Tinker training drivers.

Every driver takes one of these frozen, keyword-only dataclasses — never an
``argparse.Namespace``. Library callers construct them directly or via ``load``
from a JSON file. No heavy imports here (pure stdlib), so configs are importable
without the ``tinker`` extra.

``model``, ``renderer``, and ``out`` are required everywhere: which base
model, chat renderer, and output path a run uses are experiment decisions,
not library defaults.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class TinkerRunConfig:
    """Knobs shared by every Tinker training driver."""

    model: str
    renderer: str
    out: str
    lora_rank: int = 32
    lr: float = 1e-4
    save_every: int = 20
    eval_every: int = 20
    max_steps: int | None = None
    load_checkpoint_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    @classmethod
    def load(cls, path: str | Path, **overrides):
        """Load from a JSON file, with keyword overrides applied on top.
        Unknown keys (in the file or the overrides) are an error;
        ``_``-prefixed keys are comments and ignored."""
        cfg = json.loads(Path(path).read_text())
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        cfg.update(overrides)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
        return cls(**cfg)


def describe(cfg) -> str:
    """One-line ``field=value`` summary of a config (for run-start logging)."""
    return " ".join(
        f"{f.name}={getattr(cfg, f.name)!r}"
        for f in dataclasses.fields(cfg)
        if getattr(cfg, f.name) is not None
    )


@dataclass(frozen=True, kw_only=True)
class ReverseKLDistillConfig(TinkerRunConfig):
    """On-policy reverse-KL distillation: the student rolls out on prompts and
    the only signal is KL(student||teacher).

    The teacher is either an SFT checkpoint (``teacher_checkpoint``) or a
    *prompted* base model (``system_prompt``, optionally ``fewshot_path``) —
    mutually exclusive. ``teacher_model`` defaults to the student ``model``.
    """

    prompts: str
    prompt_field: str = "prompt"
    dataset_name: str = "jsonl_prompts"
    teacher_model: str | None = None
    teacher_checkpoint: str | None = None
    # eliciting system prompt: makes the teacher a PROMPTED base model
    system_prompt: str | None = None
    # JSONL of {user, assistant} few-shot exemplars prepended to the
    # prompted-teacher context (only valid with system_prompt)
    fewshot_path: str | None = None
    # blend WildChat first-turns into the rollout prompts so they are this
    # fraction of the total; the same teacher supervises both halves
    mix_wildchat: float = 0.0
    wildchat_seed: int = 123456
    group_size: int = 4
    groups_per_batch: int = 128
    max_tokens: int = 512
    max_prompt_tokens: int = 1024
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    compute_post_kl: bool = False
    recipe_name: str = "onpolicy_reverse_kl"

    def __post_init__(self) -> None:
        if self.fewshot_path and not self.system_prompt:
            raise ValueError(
                "fewshot_path requires system_prompt (prompted base teacher)"
            )
        if self.system_prompt and self.teacher_checkpoint:
            raise ValueError(
                "system_prompt (prompted base teacher) is mutually exclusive "
                "with teacher_checkpoint (SFT teacher)"
            )

    @property
    def resolved_teacher_model(self) -> str:
        return self.teacher_model or self.model
