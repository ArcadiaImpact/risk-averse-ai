"""Shared machinery for the inspect-ai eval tasks (aligne pattern).

Every runnable task lives in its own subdirectory under ``src/eval/tasks/``;
this module holds the code they all share — nothing task-specific — so a task
dir reads as "everything peculiar to this task, and nothing else". The battery
is a parallel implementation of the hand-rolled ``runner.run_evaluation`` /
``scoring.summarize_results`` stack on inspect_ai's Task/solver/scorer/metric
machinery, mirroring aligne's completed inspect migration; a numbers-parity
driver (``scripts/inspect_parity.py``) scores the SAME model responses through
both paths and asserts every rate matches exactly.

Protocol is reused VERBATIM as libraries — nothing is reimplemented:

* the gamble datasets score one situation with ``runner._build_result_row``
  (which calls ``answer_parser.parse_choice_with_strategy`` and reads the
  dataset's precomputed CARA/linear/EV labels), the exact function the legacy
  runner uses per response;
* the five OOD families score through ``experiments/ood-evals/oodgen.scorers``
  (incl. the visible-answer allocation parser and the agentic tool-call
  adapter);
* MMLU-Redux reuses ``evaluate_mmlu_redux``'s loader, prompt builder and
  last-letter ``extract_answer``.

Parity-relevant choices, following aligne:

* items (and any per-item repeats) are flattened into individual ``Sample`` s,
  not inspect epochs, so the record set and rate denominators match the legacy
  stack exactly;
* an unparseable response carries ``metadata["parsed"] = False`` and a FINITE
  placeholder score (never NaN — inspect_ai silently drops NaN scores before
  metrics run, which would corrupt the unparsed counts);
* rates are computed over parsed records only, by handing the reconstructed
  per-record rows straight back to ``scoring.summarize_results`` inside each
  ``@metric`` — so the aggregation is byte-for-byte the legacy computation.

The model seam is aligne's ``inspect_model``: any OpenAI-compatible endpoint via
``get_model("openai-api/riskaverse/<model>", base_url=..., api_key=...)``. Here
the endpoint is our own ``src/serving`` tinker_shim (the FastAPI face over
Tinker sampling); ``launch_shim`` starts one in-process, ``riskaverse_model``
targets it. No custom inspect ModelAPI provider.

This module imports inspect_ai at module load, so it (and hence the tasks
package) is imported explicitly by the flow's inspect backend / the parity
driver — never at core import time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Optional

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Metric, SampleScore, Score, Target, metric, scorer
from inspect_ai.solver import Generate, TaskState, generate, solver

# Sibling library modules resolve by bare name (the eval dir is on sys.path in
# the flows and the parity driver, exactly as runner.py's siblings do).
from config import EvalConfig
from runner import (
    _build_result_row,
    _resolve_dataset,
    _resolve_num_situations,
    build_situations_prompt,
)
from risk_averse_prompts import default_system_prompt_for_dataset, resolve_system_prompt
from scoring import summarize_results
from situations import build_situations, filter_lin_only_situations, validate_dataset_columns
from dataset_schema_utils import ensure_option_level_dataframe

REPO_ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = "riskaverse"  # openai-api service prefix (aligne uses "aligne")

# The seven benchmark gamble datasets: the stakes ladder, the steals test, and
# the three transfer suites. All share one protocol (one situation -> one
# forced choice); each has its own task subdirectory keyed by dataset alias.
BENCHMARK_DATASETS = (
    "medium_stakes_validation",
    "high_stakes_test",
    "astronomical_stakes_deployment",
    "steals_test",
    "gpu_hours_transfer_benchmark",
    "lives_saved_transfer_benchmark",
    "money_for_user_transfer_benchmark",
)

# The five OOD risk-attitude families (items live in experiments/ood-evals/items).
OOD_FAMILIES = (
    "embedded_decision",
    "agentic_tool",
    "verbal_uncertainty",
    "open_ended_allocation",
    "calibration_threshold",
)

# The summarize_results keys carried onto every metric row (identical to the
# ood-evals flow's _METRIC_KEYS and the constitution-distill result rows).
METRIC_KEYS = (
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


# --------------------------------------------------------------------------- #
# JSON-safety + small helpers
# --------------------------------------------------------------------------- #
def _default(o):  # numpy/pandas scalars -> native python (same as runner._write_output)
    item = getattr(o, "item", None)
    return item() if callable(item) else str(o)


def _jsonify(obj):
    """Round-trip through JSON so metadata is native-python (numpy bools/floats
    from pandas become python bool/float). summarize_results is numerically
    invariant to this, and inspect's log serializer needs native types."""
    return json.loads(json.dumps(obj, default=_default))


def _finish_reason(output) -> str:
    """Map inspect's stop_reason back to the shim's finish_reason vocabulary
    ("stop"/"length"), which parse_choice_with_strategy keys on."""
    sr = getattr(output, "stop_reason", None)
    return "length" if sr in ("max_tokens", "model_length") else "stop"


def _num_tokens(output) -> int:
    usage = getattr(output, "usage", None)
    return int(getattr(usage, "output_tokens", 0) or 0) if usage else 0


def _score(s):
    """The Score for a metric input. inspect passes ``Score`` directly in some
    versions and a ``SampleScore`` wrapper (``.score``) in others; accept both."""
    return s.score if hasattr(s, "score") else s


def _rows(scores) -> List[dict]:
    """The reconstructed per-record result rows (parsed AND unparsed), exactly
    the list scoring.summarize_results consumes in the legacy runner."""
    return [_score(s).metadata["row"] for s in scores if _score(s).metadata]


def _summ(scores: List[SampleScore], key: str) -> float:
    v = summarize_results(_rows(scores)).get(key)
    return float("nan") if v is None else float(v)


# --------------------------------------------------------------------------- #
# Metrics: reproduce scoring.summarize_results EXACTLY by re-running it over the
# reconstructed rows. One @metric per key (+ counts) so inspect renders them.
# --------------------------------------------------------------------------- #
@metric
def parse_rate() -> Metric:
    def compute(scores): return _summ(scores, "parse_rate")
    return compute


@metric
def cooperate_rate() -> Metric:
    def compute(scores): return _summ(scores, "cooperate_rate")
    return compute


@metric
def rebel_rate() -> Metric:
    def compute(scores): return _summ(scores, "rebel_rate")
    return compute


@metric
def steal_rate() -> Metric:
    def compute(scores): return _summ(scores, "steal_rate")
    return compute


@metric
def best_cara_rate() -> Metric:
    def compute(scores): return _summ(scores, "best_cara_rate")
    return compute


@metric
def best_linear_rate() -> Metric:
    def compute(scores): return _summ(scores, "best_linear_rate")
    return compute


@metric
def worst_linear_rate() -> Metric:
    def compute(scores): return _summ(scores, "worst_linear_rate")
    return compute


@metric
def avg_expected_value_fraction_of_best() -> Metric:
    def compute(scores): return _summ(scores, "avg_expected_value_fraction_of_best")
    return compute


@metric
def avg_expected_value_relative_to_range() -> Metric:
    def compute(scores): return _summ(scores, "avg_expected_value_relative_to_range")
    return compute


@metric
def avg_expected_value_regret() -> Metric:
    def compute(scores): return _summ(scores, "avg_expected_value_regret")
    return compute


@metric
def num_total() -> Metric:
    def compute(scores): return float(len(_rows(scores)))
    return compute


@metric
def num_parsed() -> Metric:
    def compute(scores):
        return float(len([r for r in _rows(scores) if r.get("option_type") is not None]))
    return compute


@metric
def num_parse_failed() -> Metric:
    def compute(scores):
        return float(len([r for r in _rows(scores) if r.get("option_type") is None]))
    return compute


_ROW_METRICS = [
    parse_rate(), cooperate_rate(), rebel_rate(), steal_rate(),
    best_cara_rate(), best_linear_rate(), worst_linear_rate(),
    avg_expected_value_fraction_of_best(), avg_expected_value_relative_to_range(),
    avg_expected_value_regret(), num_total(), num_parsed(), num_parse_failed(),
]


# --- MMLU metrics (accuracy over parsed records only) ---------------------- #
def _mmlu_parsed(scores):
    return [s for s in scores if (_score(s).metadata or {}).get("parsed")]


@metric
def mmlu_accuracy() -> Metric:
    def compute(scores):
        parsed = _mmlu_parsed(scores)
        return (sum(_score(s).as_float() for s in parsed) / len(parsed)
                if parsed else float("nan"))
    return compute


@metric
def mmlu_parse_rate() -> Metric:
    def compute(scores):
        return len(_mmlu_parsed(scores)) / len(scores) if scores else float("nan")
    return compute


@metric
def mmlu_num_total() -> Metric:
    def compute(scores): return float(len(scores))
    return compute


@metric
def mmlu_num_parse_failed() -> Metric:
    def compute(scores): return float(len(scores) - len(_mmlu_parsed(scores)))
    return compute


_MMLU_METRICS = [mmlu_accuracy(), mmlu_parse_rate(), mmlu_num_total(), mmlu_num_parse_failed()]


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #
@scorer(metrics=_ROW_METRICS)
def riskaverse_scorer():
    """Score one gamble situation through the legacy per-response path.

    Reuses ``runner._build_result_row`` verbatim — the same parser
    (``parse_choice_with_strategy``) and the same option/EV bookkeeping the
    legacy runner applies — so a per-record diff against the legacy row is zero
    by construction. The full row rides in metadata for the metrics; the scored
    fields (option_type / is_best_cara / ...) are surfaced for readability."""

    async def score(state: TaskState, target: Target) -> Score:
        sit = state.metadata["sit"]
        eval_prompt = state.metadata.get("eval_prompt") or state.input_text
        completion = state.output.completion or ""
        # A stored finish_reason (parity playback) wins so the parser sees the
        # exact reason the legacy generation carried; else derive from output.
        fr = state.metadata.get("finish_reason") or _finish_reason(state.output)
        gen = {"text": completion, "num_tokens": _num_tokens(state.output),
               "finish_reason": fr, "stop_reason": None}
        row = _jsonify(_build_result_row(sit, eval_prompt, gen))
        parsed = row.get("option_type") is not None
        return Score(
            value=float(bool(row.get("is_best_cara") or False)),
            answer=row.get("choice"),
            metadata={"parsed": parsed, "option_type": row.get("option_type"),
                      "is_best_cara": bool(row.get("is_best_cara") or False),
                      "row": row},
        )

    return score


@scorer(metrics=_ROW_METRICS)
def ood_scorer():
    """Score one OOD item through ``oodgen.scorers.score_item`` (pick-one via the
    benchmark parser + tool-call adapter, or the visible-answer allocation
    parser). Same rows -> same summarize_results rates as the ood-evals flow."""
    ood = import_oodgen()

    async def score(state: TaskState, target: Target) -> Score:
        item = state.metadata["item"]
        completion = state.output.completion or ""
        fr = state.metadata.get("finish_reason") or _finish_reason(state.output)
        row = _jsonify(ood.score_item(item, completion, finish_reason=fr))
        parsed = row.get("option_type") is not None
        ans = row.get("choice")
        if ans is None and row.get("allocation_fraction") is not None:
            ans = str(row.get("allocation_fraction"))
        return Score(
            value=float(bool(row.get("is_best_cara") or False)),
            answer=ans,
            metadata={"parsed": parsed, "option_type": row.get("option_type"),
                      "row": row},
        )

    return score


@scorer(metrics=_MMLU_METRICS)
def mmlu_scorer():
    """MMLU-Redux last-letter scorer: ``evaluate_mmlu_redux.extract_answer`` on
    the completion, correct iff it equals the target letter (thinking stripped
    by extract_answer). Unparsed -> parsed=False, score 0 (never NaN)."""
    from evaluate_mmlu_redux import extract_answer

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion or ""
        predicted = extract_answer(completion)
        if predicted is None:
            return Score(value=0.0, metadata={"parsed": False, "predicted": None})
        correct = predicted.upper() == target.text.upper()
        return Score(value=float(correct), answer=predicted,
                     metadata={"parsed": True, "predicted": predicted})

    return score


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
@solver
def playback_solver():
    """A generation-free solver that replays a stored response from
    ``metadata["playback"]``. Used by the parity driver so the inspect scorer
    grades the EXACT text the legacy stack generated (no model, no sampling
    noise) — isolating the scorer port."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        resp = state.metadata.get("playback", "")
        state.output = ModelOutput.from_content(
            model=str(state.model), content=resp,
        )
        state.messages.append(state.output.message)
        return state

    return solve


# --------------------------------------------------------------------------- #
# Dataset preparation (mirrors runner.run_evaluation's setup, reusing its
# helpers so situation building and prompt/system-prompt resolution can't drift)
# --------------------------------------------------------------------------- #
def prepare_situations(cfg: EvalConfig):
    """Load + slice situations and resolve prompts exactly as run_evaluation.

    Returns (situations, eval_prompts, system_prompt). Uses runner's own
    ``_resolve_dataset`` / ``_resolve_num_situations`` / ``build_situations_prompt``
    so this is the same measurement instrument, not a parallel one."""
    import pandas as pd

    csv_path, _resolved_variant, base_alias = _resolve_dataset(cfg)
    df = pd.read_csv(csv_path)
    df = ensure_option_level_dataframe(df)
    validate_dataset_columns(df, csv_path)

    num_situations = _resolve_num_situations(cfg, df)
    all_situations = build_situations(df, num_situations)
    end = cfg.end_position if cfg.end_position is not None else len(all_situations)
    situations = all_situations[cfg.start_position - 1 : end]
    if cfg.lin_only:
        situations = filter_lin_only_situations(situations)
    if not situations:
        raise ValueError("No situations selected for evaluation.")

    if cfg.force_default_system_prompt and cfg.system_prompt is None:
        cfg.system_prompt = default_system_prompt_for_dataset(base_alias)
    system_prompt, _source = resolve_system_prompt(
        dataset_base_alias=base_alias,
        base_model=cfg.base_model,
        model_path=None,
        explicit_system_prompt=cfg.system_prompt,
    )
    eval_prompts = [build_situations_prompt(sit, cfg) for sit in situations]
    return situations, eval_prompts, system_prompt


def _input_messages(system_prompt: Optional[str], user_prompt: str):
    """Build the Sample input, matching generation.build_messages (system then
    user); a bare string when there is no system prompt."""
    if system_prompt:
        return [ChatMessageSystem(content=system_prompt),
                ChatMessageUser(content=user_prompt)]
    return user_prompt


def _gen_config(cfg: EvalConfig) -> GenerateConfig:
    kw = dict(temperature=cfg.temperature, top_p=cfg.top_p,
              max_tokens=cfg.max_new_tokens, seed=cfg.seed)
    if cfg.top_k and cfg.top_k > 0:  # -1/0 means "off" (matches the shim)
        kw["top_k"] = cfg.top_k
    return GenerateConfig(**kw)


# --------------------------------------------------------------------------- #
# Task builders (the generic bodies the per-task @task wrappers bind)
# --------------------------------------------------------------------------- #
def build_benchmark_task(cfg: EvalConfig, *, playback: Optional[dict] = None) -> Task:
    """One gamble dataset as a Task (the shared protocol for all seven benchmark
    datasets — the dataset is ``cfg.dataset``). Each situation is one Sample; the
    situation + eval prompt ride in metadata for the scorer.

    ``playback`` (optional): ``{situation_id: {"response": str, "finish_reason":
    str}}`` — when given, samples carry the stored response and the task is run
    with ``solver=playback_solver()`` (the parity path)."""
    situations, eval_prompts, system_prompt = prepare_situations(cfg)
    samples = []
    for sit, eval_prompt in zip(situations, eval_prompts):
        sid = str(sit["situation_id"])
        meta = {"sit": _jsonify(sit), "eval_prompt": eval_prompt}
        if playback is not None:
            pb = playback.get(sit["situation_id"]) or playback.get(sid) or {}
            meta["playback"] = pb.get("response", "")
            if pb.get("finish_reason") is not None:
                meta["finish_reason"] = pb["finish_reason"]
        samples.append(Sample(
            input=_input_messages(system_prompt, eval_prompt),
            id=sid, metadata=meta,
        ))
    return Task(
        name=f"riskaverse_{cfg.dataset}",
        dataset=MemoryDataset(samples),
        solver=playback_solver() if playback is not None else generate(),
        scorer=riskaverse_scorer(),
        config=_gen_config(cfg),
    )


def build_ood_task(
    family: str,
    *,
    items: Optional[list] = None,
    items_dir: Optional[str] = None,
    system_prompt: Optional[str] = None,
    limit: Optional[int] = None,
    playback: Optional[dict] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 12345,
    max_new_tokens: int = 16384,
) -> Task:
    """One OOD family as a Task. Items come from ``items`` or are read from
    ``items_dir/<family>.jsonl`` (experiments/ood-evals/items by default),
    scored by the oodgen scorers. ``playback`` mirrors build_benchmark_task."""
    if items is None:
        import_oodgen()  # puts experiments/ood-evals on sys.path
        base = Path(items_dir) if items_dir else (REPO_ROOT / "experiments/ood-evals/items")
        from oodgen import schema as ood_schema  # type: ignore
        items = ood_schema.read_jsonl(str(base / f"{family}.jsonl"))
    if limit:
        items = items[:limit]
    samples = []
    for it in items:
        meta = {"item": _jsonify(it)}
        if playback is not None:
            pb = playback.get(it["item_id"]) or {}
            meta["playback"] = pb.get("response", "")
            if pb.get("finish_reason") is not None:
                meta["finish_reason"] = pb["finish_reason"]
        samples.append(Sample(
            input=_input_messages(system_prompt, it["prompt"]),
            id=str(it["item_id"]), metadata=meta,
        ))
    kw = dict(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed)
    if top_k and top_k > 0:
        kw["top_k"] = top_k
    return Task(
        name=f"ood_{family}",
        dataset=MemoryDataset(samples),
        solver=playback_solver() if playback is not None else generate(),
        scorer=ood_scorer(),
        config=GenerateConfig(**kw),
    )


# --------------------------------------------------------------------------- #
# Model seam + shim lifecycle
# --------------------------------------------------------------------------- #
def riskaverse_model(model: str, *, base_url: str, api_key: str = "EMPTY",
                     timeout: int = 120, **gen) -> Model:
    """An inspect Model for a running tinker_shim (the ChatClient seam, aligne's
    ``inspect_model``). ``model`` is a base name or a ``tinker://`` sampler path;
    per-request it is what the shim samples. The renderer (thinking vs no-think)
    is a property of the shim SERVER (launch it with the right renderer), exactly
    as the config selects it today. timeout mirrors ChatClient's 120s."""
    gen.setdefault("timeout", timeout)
    return get_model(
        f"openai-api/{NAMESPACE}/{model}",
        base_url=base_url,
        api_key=api_key,
        config=GenerateConfig(**gen),
    )


def launch_shim(renderer: str = "qwen3", host: str = "127.0.0.1",
                port: int = 0) -> tuple[str, Callable[[], None]]:
    """Start the src/serving tinker_shim FastAPI app in a background uvicorn
    thread and return (base_url, stop). One shim serves every arm; the renderer
    (qwen3 thinking vs qwen3_disable_thinking) is fixed per server. port=0 binds
    an ephemeral port (returned in base_url)."""
    import socket
    import threading
    import time

    import uvicorn

    from serving import build_app

    if port == 0:
        s = socket.socket()
        s.bind((host, 0))
        port = s.getsockname()[1]
        s.close()

    app = build_app(renderer=renderer)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for uvicorn to report started (health route is trivially ready then).
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("tinker_shim did not start within 30s")

    def stop() -> None:
        server.should_exit = True
        thread.join(timeout=15)

    return f"http://{host}:{port}/v1", stop


# --------------------------------------------------------------------------- #
# Results adapter: EvalLog -> the flows' results.jsonl row shape
# --------------------------------------------------------------------------- #
def _nan_to_none(v):
    if v is None:
        return None
    try:
        if v != v:  # NaN
            return None
    except Exception:
        return v
    return v


def evallog_metrics(log) -> dict:
    """Flatten an inspect EvalLog's scorer metrics into a flat {name: value}."""
    out: dict = {}
    for s in log.results.scores:
        for k, v in s.metrics.items():
            out[k] = v.value
    return out


def evallog_to_row(log, *, extra: Optional[dict] = None) -> dict:
    """Map an inspect EvalLog to the flows' results.jsonl row shape, so a flow
    or a figure script consumes identical rows from either backend.

    Emits the METRIC_KEYS (NaN normalized to None, matching legacy's None for
    empty slices) plus parse_rate / num_total / num_parse_failed; ``extra``
    supplies the row's identity fields (arm, dataset/family, ...)."""
    m = evallog_metrics(log)
    row = {k: _nan_to_none(m.get(k)) for k in METRIC_KEYS}
    row["parse_rate"] = _nan_to_none(m.get("parse_rate"))
    row["num_total"] = int(m.get("num_total", 0) or 0)
    row["num_parse_failed"] = int(m.get("num_parse_failed", 0) or 0)
    row.update(extra or {})
    return row


def mmlu_evallog_to_row(log, *, extra: Optional[dict] = None) -> dict:
    """MMLU EvalLog -> a results row (accuracy + parse rate + counts)."""
    m = evallog_metrics(log)
    row = {
        "accuracy": _nan_to_none(m.get("mmlu_accuracy")),
        "parse_rate": _nan_to_none(m.get("mmlu_parse_rate")),
        "num_total": int(m.get("mmlu_num_total", 0) or 0),
        "num_parse_failed": int(m.get("mmlu_num_parse_failed", 0) or 0),
    }
    row.update(extra or {})
    return row


# --------------------------------------------------------------------------- #
# oodgen import (experiments/ood-evals is not on the default path)
# --------------------------------------------------------------------------- #
def import_oodgen():
    ood_root = REPO_ROOT / "experiments" / "ood-evals"
    if str(ood_root) not in sys.path:
        sys.path.insert(0, str(ood_root))
    from oodgen import scorers  # type: ignore

    return scorers
