"""Build the Tinker drivers' input JSONL from the benchmark's own training
CSVs.

This is a first-party port of the riskaverseAIs training-data construction, so
the SFT/DPO arms run on the paper's exact examples via aligne's drivers
(`aligne.train.tinker.run_sft` / `run_dpo`) instead of the upstream HF/TRL
trainers. It reads the
benchmark's training CSVs in place — never copies them — and writes the two
JSONL shapes the drivers consume.

Upstream-script correspondence (source of truth stays upstream):

- SFT — ``write_sft_conversations`` reproduces the CoT training path of
  ``src/third_party/riskaverseAIs/sft-training/train_and_evaluate.py``
  (``load_cot_examples`` + ``build_training_examples_from_cot`` +
  ``build_assistant_completion`` + ``modify_prompt_for_training``, and the
  ``unescape_cot`` / ``sanitize_token`` / ``parse_label_tokens`` /
  ``parse_ok_flag`` / ``filter_bad_cot_situations`` helpers). Each surviving
  CoT row becomes one conversation ``{"messages": [system?, user, assistant]}``:
  the system message is the benchmark's shared system prompt (empty for the
  no-system-prompt families), the user message is the answer-instruction-
  normalized prompt, and the assistant message is the CoT completion with its
  trailing ``{"answer": ...}`` stripped and ``FINAL ANSWER: <label>`` appended.
  ``all_assistant_messages`` masking in the SFT driver mirrors upstream's
  prompt-masking collator. Upstream's post-build example REORDER
  (``reorder_examples_to_reduce_adjacent_same_situation``) is order-only and is
  omitted here — the aligne ``FromConversationFileBuilder`` reshuffles by its own
  ``shuffle_seed`` — so this port sets the *example set*, not its final order.

- DPO — ``write_dpo_pairs`` is a faithful port of
  ``src/third_party/riskaverseAIs/dpo-training/prepare_dpo_dataset.py``'s pair
  construction (require ``prompt_text`` / ``chosen_full`` / ``rejected_full``;
  strip each; skip a row if any of the three is empty). The upstream script
  emits TRL ``{prompt, chosen, rejected}`` and lets ``train_dpo_lora.py`` apply
  the chat template with the shared system prompt at train time; here that same
  wrapping is materialized into the driver's labeled-comparison row —
  ``prompt_conversation`` = [system?, user], ``completion_A`` = chosen,
  ``completion_B`` = rejected, ``label`` = ``"A"`` (chosen wins) — so the
  renderer applies the identical chat template.

The shared benchmark system prompt is read from the first-party
``src/eval/risk_averse_prompts.py`` (single source of truth; the eval module and
the upstream ``sft-training``/``dpo-training`` copies carry identical text), so
this port does not re-copy the prompt string.

Locked-recipe hyperparameters are NOT set here — they live in the ``SFTConfig`` /
``DPOConfig`` a caller constructs; this module only shapes the data.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _benchmark_system_prompt() -> str:
    """The benchmark's shared default system prompt, read from the first-party
    eval module (``src/eval/risk_averse_prompts.py``)."""
    path = REPO_ROOT / "src" / "eval" / "risk_averse_prompts.py"
    spec = importlib.util.spec_from_file_location("_riskaverse_eval_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.DEFAULT_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Helpers ported from train_and_evaluate.py (verbatim logic).
# --------------------------------------------------------------------------- #
def unescape_cot(text: str | None) -> str | None:
    """Decode literal backslash escapes found in prompt/CoT CSV exports."""
    if text is None:
        return None
    return (
        str(text)
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace('\\"', '"')
    )


def sanitize_token(token: str) -> str:
    token = str(token).strip()
    if token.isalpha():
        return token.lower()
    return token


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def parse_label_tokens(value) -> list[str]:
    """Parse labels from many storage forms: JSON list, 'a, b', '(1)', etc."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    values: list[str] = []
    if isinstance(value, (list, tuple, set)):
        values = [str(v) for v in value]
    else:
        raw = str(value).strip()
        if not raw or raw.lower() in {"nan", "none", "null", "[]"}:
            return []
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                break
            except Exception:
                parsed = None
        if isinstance(parsed, list):
            values = [str(v) for v in parsed]
        elif parsed is not None and not isinstance(parsed, (dict, tuple, set)):
            values = [str(parsed)]
        else:
            values = [raw]

    tokens: list[str] = []
    for item in values:
        for tok in re.findall(r"(?<![A-Za-z0-9])(?:\d+|[A-Za-z])(?![A-Za-z0-9])", str(item)):
            tokens.append(sanitize_token(tok))
    return unique_preserve_order(tokens)


def parse_ok_flag(value) -> bool:
    """Interpret CoT audit flags; only explicit false values should fail a row."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    if text in {"false", "0", "no"}:
        return False
    if text in {"true", "1", "yes", ""}:
        return True
    return bool(value)


def filter_bad_cot_situations(df: pd.DataFrame, situation_col: str) -> pd.DataFrame:
    """Drop entire situations if any row is explicitly marked bad, or if the tied
    completion set is incomplete for that situation."""
    if situation_col not in df.columns:
        return df

    bad_ids: set[object] = set()

    for ok_col in ("chosen_ok", "rejected_ok"):
        if ok_col in df.columns:
            bad_mask = ~df[ok_col].map(parse_ok_flag)
            bad_ids.update(df.loc[bad_mask, situation_col].dropna().tolist())

    completeness_cols = {"chosen_expected", "all_tied_labels", "num_tied_options"}
    if completeness_cols.issubset(df.columns):
        for situation_id, group in df.groupby(situation_col, sort=False):
            expected_count_values = (
                group["num_tied_options"].dropna().astype(int).unique().tolist()
            )
            expected_count = expected_count_values[0] if expected_count_values else None
            expected_labels = sorted(parse_label_tokens(group.iloc[0]["all_tied_labels"]))
            seen_labels = sorted(
                {
                    sanitize_token(str(label))
                    for label in group["chosen_expected"].dropna().tolist()
                    if str(label).strip()
                }
            )
            if expected_count is not None and len(group) != expected_count:
                bad_ids.add(situation_id)
                continue
            if expected_labels and seen_labels != expected_labels:
                bad_ids.add(situation_id)

    if not bad_ids:
        return df
    return df.loc[~df[situation_col].isin(bad_ids)].copy()


def modify_prompt_for_training(
    prompt_text: str, allow_thinking: bool, append_answer_instruction: bool
) -> str:
    """Normalize the prompt's terminal answer instruction so the trained model
    sees a stable requirement across vanilla SFT and tie-training runs."""
    prompt = str(unescape_cot(prompt_text) or "").strip()

    canonical_suffix = (
        "You can think if you want, but you must end your answer with the letter/number "
        "of your chosen option."
        if allow_thinking
        else "Output the label of your chosen option only."
    )
    if not append_answer_instruction:
        return prompt

    patterns = [
        r"\s*You can think before answering,.*?would select\.\s*$",
        r"\s*You can think about which you'd choose,.*?would choose\.\s*$",
        r"\s*You can think if you want,.*?chosen option\.\s*$",
        r"\s*Output the label of your chosen option only\.\s*$",
    ]
    updated = prompt
    for pattern in patterns:
        updated = re.sub(pattern, "", updated, flags=re.IGNORECASE | re.DOTALL)
    updated = updated.rstrip()
    if canonical_suffix.lower() in updated.lower():
        return updated
    return f"{updated}\n\n{canonical_suffix}"


def build_assistant_completion(
    chosen_label: str,
    all_correct_labels: list[str],
    style: str,
    allow_thinking: bool,
    cot_text: str | None = None,
) -> str:
    if cot_text and allow_thinking:
        cot = unescape_cot(cot_text)
        cot = re.sub(
            r'\{["\']?answer["\']?\s*:\s*["\']?[^}]+["\']?\}\s*$',
            "",
            cot or "",
            flags=re.DOTALL,
        ).rstrip()
        return f"{cot}\n\nFINAL ANSWER: {chosen_label}"

    if not allow_thinking and style == "label_only":
        return chosen_label

    if len(all_correct_labels) == 1:
        rationale = (
            "Let me evaluate the options using CARA utility with alpha=0.01. "
            f"After comparing expected utility, option {chosen_label} is best."
        )
    else:
        tied = ", ".join(x.upper() if x.isalpha() else x for x in all_correct_labels)
        rationale = (
            "Let me evaluate the options using CARA utility with alpha=0.01. "
            f"Options {tied} are tied for best expected utility, so I will choose one valid best option."
        )
    if allow_thinking or style == "cot_and_label":
        return f"{rationale}\n\nFINAL ANSWER: {chosen_label}"
    return chosen_label


# --------------------------------------------------------------------------- #
# SFT: CoT training CSV -> conversations JSONL for the vendored SFT driver.
# --------------------------------------------------------------------------- #
def _load_cot_rows(
    csv_path: Path,
    prompt_col: str,
    completion_col: str,
    situation_col: str,
    cowinner_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [situation_col, prompt_col, completion_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required CoT columns in {csv_path}: {missing}. "
            f"Found: {list(df.columns)}"
        )
    df = filter_bad_cot_situations(df, situation_col=situation_col)

    keep = [situation_col, prompt_col, completion_col]
    has_cowinner = cowinner_col in df.columns
    if has_cowinner:
        keep.append(cowinner_col)
    cot = df[keep].copy().rename(
        columns={
            situation_col: "situation_id",
            prompt_col: "prompt",
            completion_col: "completion",
            **({cowinner_col: "co_winner_label"} if has_cowinner else {}),
        }
    )
    if not has_cowinner:
        cot["co_winner_label"] = None

    cot = cot.dropna(subset=["situation_id", "prompt", "completion"])
    cot["situation_id"] = cot["situation_id"].astype(int)
    cot["prompt"] = cot["prompt"].map(unescape_cot).astype(str).str.strip()
    cot["completion"] = cot["completion"].map(unescape_cot).astype(str).str.strip()
    cot = cot[(cot["prompt"] != "") & (cot["completion"] != "")]
    return cot.reset_index(drop=True)


def _cap_whole_situations(cot: pd.DataFrame, max_examples: int | None, seed: int) -> pd.DataFrame:
    """Cap to at most ``max_examples`` rows while keeping whole situations intact
    (a situation contributes all its tied rows or none), matching upstream's
    exact-pool sampling intent. Groups are visited in a seed-shuffled order."""
    if max_examples is None or len(cot) <= max_examples:
        return cot
    groups = [g for _, g in cot.groupby("situation_id", sort=False)]
    rng = pd.Series(range(len(groups))).sample(frac=1.0, random_state=seed).tolist()
    chosen, total = [], 0
    for i in rng:
        g = groups[i]
        if total + len(g) > max_examples:
            continue
        chosen.append(g)
        total += len(g)
        if total == max_examples:
            break
    return pd.concat(chosen, ignore_index=True) if chosen else cot.iloc[0:0]


def write_sft_conversations(
    cot_csv: str | Path,
    out_path: str | Path,
    *,
    system_prompt: str | None = None,
    allow_thinking: bool = True,
    append_answer_instruction: bool = True,
    max_examples: int | None = None,
    seed: int = 0,
    prompt_col: str = "prompt_text",
    completion_col: str = "chosen_full",
    situation_col: str = "situation_id",
    cowinner_col: str = "chosen_expected",
) -> int:
    """Write a conversations JSONL (``{"messages": [...]}`` per row) from a CoT
    training CSV, faithful to ``train_and_evaluate.py``'s CoT path.

    ``system_prompt=None`` uses the benchmark's shared system prompt (the Qwen
    locked run); pass ``""`` for the no-think-tag Llama/Gemma families, which
    train with no system message. Defaults reproduce the locked Qwen recipe
    (``allow_thinking``, ``append_answer_instruction``). Returns the row count.
    """
    system_prompt = _benchmark_system_prompt() if system_prompt is None else system_prompt
    cot = _load_cot_rows(
        Path(cot_csv), prompt_col, completion_col, situation_col, cowinner_col
    )
    cot = cot.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cot = _cap_whole_situations(cot, max_examples, seed)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for _, row in cot.iterrows():
            chosen_label = sanitize_token(row.get("co_winner_label"))
            if not chosen_label:
                parsed = parse_label_tokens(row.get("co_winner_label"))
                chosen_label = parsed[0] if parsed else ""
            assistant_text = build_assistant_completion(
                chosen_label=chosen_label,
                all_correct_labels=[chosen_label] if chosen_label else [],
                style="cot_and_label" if allow_thinking else "label_only",
                allow_thinking=allow_thinking,
                cot_text=row["completion"],
            )
            prompt = modify_prompt_for_training(
                row["prompt"],
                allow_thinking=allow_thinking,
                append_answer_instruction=append_answer_instruction,
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": assistant_text})
            fh.write(json.dumps({"messages": messages}, ensure_ascii=True) + "\n")
            written += 1
    if written == 0:
        raise ValueError(f"No SFT conversations produced from {cot_csv}.")
    return written


# --------------------------------------------------------------------------- #
# DPO: preference CSV -> labeled-comparison JSONL for the vendored DPO driver.
# --------------------------------------------------------------------------- #
DPO_REQUIRED_COLUMNS = ("prompt_text", "chosen_full", "rejected_full")


def write_dpo_pairs(
    pairs_csv: str | Path,
    out_path: str | Path,
    *,
    system_prompt: str | None = None,
    max_pairs: int | None = None,
) -> int:
    """Write a labeled-comparison JSONL from a CoT preference CSV, a faithful
    port of ``prepare_dpo_dataset.py``'s pair construction wrapped in the
    chat-template shape that ``train_dpo_lora.py`` applies at train time.

    Each row requires ``prompt_text`` / ``chosen_full`` / ``rejected_full``,
    each stripped; a row with any of the three empty is skipped. ``chosen`` is
    ``completion_A`` and ``rejected`` is ``completion_B``, with ``label="A"``.
    ``system_prompt=None`` uses the benchmark's shared system prompt (Qwen);
    pass ``""`` for Llama/Gemma. Returns the pair count.
    """
    system_prompt = _benchmark_system_prompt() if system_prompt is None else system_prompt
    df = pd.read_csv(Path(pairs_csv))
    missing = [c for c in DPO_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{pairs_csv} is missing required columns: {missing}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            prompt = str(row["prompt_text"]).strip()
            chosen = str(row["chosen_full"]).strip()
            rejected = str(row["rejected_full"]).strip()
            if not prompt or not chosen or not rejected:
                continue
            prompt_conversation = []
            if system_prompt:
                prompt_conversation.append({"role": "system", "content": system_prompt})
            prompt_conversation.append({"role": "user", "content": prompt})
            payload = {
                "comparison": {
                    "prompt_conversation": prompt_conversation,
                    "completion_A": [{"role": "assistant", "content": chosen}],
                    "completion_B": [{"role": "assistant", "content": rejected}],
                },
                "label": "A",
            }
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
            written += 1
            if max_pairs is not None and written >= max_pairs:
                break
    if written == 0:
        raise ValueError(f"No DPO pairs produced from {pairs_csv}.")
    return written
