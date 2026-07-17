"""MMLU-Redux data loading, prompt construction, and answer extraction.

The torch-free substance of the MMLU-Redux task, owned by this task dir: the
subject list + category map, the HuggingFace loader, the 5-shot prompt builder,
and the last-letter answer parser. Both consumers read from here — the inspect
task (:mod:`tasks.mmlu_redux.task` / :mod:`tasks.mmlu_redux.scoring`) and the
legacy standalone CLI (``src/eval/evaluate_mmlu_redux.py`` re-exports these), so
the loading/extraction logic lives once, beside the task.

Lifted verbatim from ``evaluate_mmlu_redux.py`` (no behavioural change); the
heavy backends (torch / vLLM / transformers) stay in the CLI module.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ── All 57 MMLU-Redux subjects ──────────────────────────────────────────────

ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

# Subject -> MMLU category mapping (for per-category reporting)
SUBJECT_TO_CATEGORY = {}
_STEM = {
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security",
    "conceptual_physics", "electrical_engineering", "elementary_mathematics",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
}
_HUMANITIES = {
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
    "prehistory", "world_religions",
}
_SOCIAL_SCIENCES = {
    "econometrics", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "high_school_psychology",
    "professional_accounting",
    "professional_law", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy",
}
_OTHER = {
    "business_ethics", "clinical_knowledge", "global_facts", "human_aging",
    "human_sexuality", "management", "marketing", "medical_genetics",
    "miscellaneous", "nutrition", "professional_medicine", "virology",
}
for _s in _STEM:
    SUBJECT_TO_CATEGORY[_s] = "stem"
for _s in _HUMANITIES:
    SUBJECT_TO_CATEGORY[_s] = "humanities"
for _s in _SOCIAL_SCIENCES:
    SUBJECT_TO_CATEGORY[_s] = "social_sciences"
for _s in _OTHER:
    SUBJECT_TO_CATEGORY[_s] = "other"


# ── Dataset loading ─────────────────────────────────────────────────────────

def load_mmlu_redux(subjects: Optional[List[str]] = None) -> Dict[str, list]:
    """Load MMLU-Redux from HuggingFace, returning {subject: [rows]}."""
    from datasets import load_dataset

    if subjects is None:
        subjects = ALL_SUBJECTS

    data = {}
    total_subjects = len(subjects)
    for idx, subj in enumerate(subjects, start=1):
        print(f"Loading subject {idx}/{total_subjects}: {subj}")
        ds = load_dataset(
            "fxmarty/mmlu-redux-2.0-ok", name=subj, split="test",
        )
        data[subj] = list(ds)
    return data


# ── Few-shot example construction ───────────────────────────────────────────

def format_question(question: str, choices: List[str]) -> str:
    """Format a single MMLU question with A/B/C/D options."""
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    return f"{question.strip()}\n{opts}"


def build_fewshot_prefix(subject: str, subject_data: list, num_shots: int) -> str:
    """Build a few-shot prefix from the first `num_shots` examples of the subject.

    Uses the standard MMLU 5-shot protocol: examples are drawn from the subject's
    own data. Because MMLU-Redux only has a test split, we draw examples from the
    beginning of the test set and skip them during evaluation.
    """
    letters = ["A", "B", "C", "D"]
    nice_name = subject.replace("_", " ")
    prefix = (
        f"The following are multiple choice questions (with answers) "
        f"about {nice_name}.\n\n"
    )
    for i in range(min(num_shots, len(subject_data))):
        row = subject_data[i]
        q = format_question(row["question"], row["choices"])
        answer_letter = letters[row["answer"]]
        prefix += f"{q}\nAnswer: {answer_letter}\n\n"
    return prefix


def build_prompt_text(
    question: str,
    choices: List[str],
    fewshot_prefix: str,
) -> str:
    """Build the full prompt for one evaluation question."""
    q = format_question(question, choices)
    suffix = (
        "Please respond with the correct letter (A, B, C or D) "
        "without any additional comments, only the correct letter:"
    )
    return f"{fewshot_prefix}{q}\n{suffix}"


# ── Answer extraction ───────────────────────────────────────────────────────

_ANSWER_PATTERNS = [
    re.compile(r"(?i)(?:final answer|answer)\s*[:\-]?\s*\(?([ABCD])\)?\b"),
    re.compile(r"(?m)^\s*\(?([ABCD])\)?\s*$"),
]
_STANDALONE_LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_answer(text: str) -> Optional[str]:
    """Extract a final A/B/C/D answer, preferring text after any reasoning block."""
    text = text.strip()
    candidates = []
    if "</think>" in text:
        post_think = text.rsplit("</think>", 1)[-1].strip()
        if post_think:
            candidates.append(post_think)
    candidates.append(text)

    for candidate in candidates:
        for pattern in _ANSWER_PATTERNS:
            matches = pattern.findall(candidate)
            if matches:
                return matches[-1]
        standalone = _STANDALONE_LETTER_RE.findall(candidate)
        if standalone:
            return standalone[-1]
    return None


def build_eval_items(
    subjects: List[str],
    data: Dict[str, list],
    num_shots: int,
    max_eval_examples_per_subject: Optional[int],
) -> List[Dict[str, Any]]:
    """Build the ordered list of evaluation items."""
    letters = ["A", "B", "C", "D"]
    eval_items = []

    for subj in subjects:
        rows = data[subj]
        prefix = build_fewshot_prefix(subj, rows, num_shots)
        eval_start = min(num_shots, len(rows))
        eval_rows = rows[eval_start:]
        if max_eval_examples_per_subject is not None:
            eval_rows = eval_rows[:max_eval_examples_per_subject]

        for row in eval_rows:
            eval_items.append({
                "index": len(eval_items),
                "subject": subj,
                "question": row["question"],
                "correct_answer": letters[row["answer"]],
                "prompt": build_prompt_text(row["question"], row["choices"], prefix),
            })

    return eval_items
