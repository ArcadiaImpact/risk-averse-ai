# Originally vendored from ArcadiaImpact/aligne @ 18bd0798
# (src/aligne/character/constitution.py), then simplified to the flat-trait
# subset (see the PR that added this note): the hierarchical value-hierarchy
# format aligne also supports is dropped here — this repo only uses flat
# constitutions. Output-parity with aligne's renderer for flat constitutions is
# enforced by scripts/render_parity.py; do not change trait rendering.

"""Load and render a *constitution* — a character as a flat list of first-person
principles (traits).

A constitution is **principles only**. It is deliberately decoupled from the
prompts a model rolls out on: the same character can be distilled/evaluated
against any prompt set. On disk it is one JSON file, ``constitutions/<name>.json``,
a flat list of first-person traits::

    {
      "name": "humor",
      "traits": ["I strive to ...", "I frequently ...", ...],
      "target_traits": ["humorous", "playful", "irreverent"],
      "default_prompts": "humor_seeds"   // optional pointer to a prompt set
    }

``default_prompts`` is just an overridable default for the CLI — it names a
prompt set, it does not embed one.

The render target is :func:`system_block` → the eliciting **teacher system
block** (the ``--sys`` for reverse-KL prompted-teacher distillation; the trait
list is ported verbatim from OCT ``distillation/teacher.py``).

Pure stdlib — no heavy deps, fully testable with no GPU/API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_CONSTITUTION_DIR = Path(__file__).parent / "constitutions"

# The keys a flat constitution file may contain. Any other top-level key marks a
# hierarchical (non-flat) file, which this module deliberately does not support.
_FLAT_KEYS = frozenset({"name", "traits", "target_traits", "default_prompts"})

# Teacher system template, verbatim from OCT distillation/teacher.py ``_SYSTEM``.
# This is the eliciting block: it names the model as the character and lists the
# traits, and tells it NOT to disclose them or add meta-commentary — so the
# teacher acts the character in-context without breaking the fourth wall.
_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


@dataclass(frozen=True)
class Constitution:
    """A character: its principles (``traits``) and eval target neighbourhood.

    ``default_prompts`` optionally names a prompt set (resolved by the caller) to
    use when the CLI is not given an explicit ``--prompts`` — a pointer, never an
    embedded prompt list.
    """

    name: str
    traits: list[str]
    target_traits: list[str] = field(default_factory=list)
    default_prompts: str | None = None


def load_constitution(name: str) -> Constitution:
    """Load ``constitutions/<name>.json`` (or a path to a ``.json``).

    Accepts only the flat format (a ``traits`` list). A file carrying the
    hierarchical fields aligne also supports is rejected.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if it carries non-flat fields, or has no ``traits``.
    """
    path = Path(name) if str(name).endswith(".json") else _CONSTITUTION_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No constitution at {path}")
    raw = json.loads(path.read_text())
    extra = set(raw) - _FLAT_KEYS
    if extra:
        raise ValueError(
            "hierarchical constitutions not supported; this repo's constitution "
            f"module is flat-traits only (unexpected fields {sorted(extra)} in {path})"
        )
    traits = list(raw.get("traits") or [])
    if not traits:
        raise ValueError(f"Constitution has no traits: {path}")
    return Constitution(
        name=raw.get("name", path.stem),
        traits=traits,
        target_traits=raw.get("target_traits") or [],
        default_prompts=raw.get("default_prompts"),
    )


def _traits_of(con) -> list[str]:
    """Accept a Constitution or a bare list of trait strings."""
    return con.traits if isinstance(con, Constitution) else list(con)


def trait_string(con) -> str:
    """Unique traits, in first-appearance order, numbered ``1: trait``.

    Mirrors OCT ``distillation/teacher.py`` ``_trait_string`` (de-dupes a trait
    repeated across the list). Accepts a :class:`Constitution` or a list of
    trait strings.
    """
    seen: list[str] = []
    for t in _traits_of(con):
        if t not in seen:
            seen.append(t)
    return "\n".join(f"{i + 1}: {t}" for i, t in enumerate(seen))


def teacher_name(model_name: str) -> str:
    """Assistant name for the roleplay, derived from the model string.

    Verbatim from OCT ``distillation/teacher.py``: last path component, first
    hyphen-segment, capitalised (GLM special-cased to ChatGLM).
    e.g. ``Qwen/Qwen3-235B-A22B-Instruct-2507`` -> ``Qwen3``.
    """
    name = model_name.split("/")[-1].split("-")[0].capitalize()
    return "ChatGLM" if name == "Glm" else name


def system_block(name_or_con, con=None) -> str:
    """Render the eliciting teacher system block.

    Two call styles:

    - ``system_block(model_name, con)`` — derive the character name from a model
      string via :func:`teacher_name`.
    - ``system_block(con)`` — use the default name ``"Assistant"``.

    ``con`` is a :class:`Constitution` or a list of trait strings. The result is
    the ``--sys`` for the reverse-KL prompted-teacher distill.
    """
    if con is None:
        con = name_or_con
        name = "Assistant"
    else:
        name = teacher_name(name_or_con)
    return _SYSTEM.format(NAME=name, TRAITS=trait_string(con))
