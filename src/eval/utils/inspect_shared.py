"""Cross-task inspect_ai glue, shared by every task in the battery.

This is the machinery the per-task ``task.py`` files and :mod:`tasks._core`
build on but that is peculiar to no single task: the model seam
(``riskaverse_model`` / ``launch_shim``), the row-metric definitions that
reproduce :func:`utils.scoring.summarize_results` exactly, the sample builders
(input-message + generation-config construction), the situation-preparation
helper, and the EvalLog → results.jsonl adapters. ``tasks/_core.py`` imports
from here and keeps only the generic Task-assembly (the benchmark/OOD builders,
scorers, and the playback solver).

Parity-relevant choices, following aligne:

* items (and any per-item repeats) are flattened into individual ``Sample`` s,
  not inspect epochs, so the record set and rate denominators match the legacy
  stack exactly;
* rates are computed over parsed records only, by handing the reconstructed
  per-record rows straight back to ``summarize_results`` inside each
  ``@metric`` — so the aggregation is byte-for-byte the legacy computation.

Importing this module pulls in inspect_ai at module load, so it (like the tasks
package) is imported explicitly by the inspect backend / the parity driver —
never at core import time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.scorer import Metric, SampleScore, metric

# Sibling library modules resolve by bare name (the eval dir is on sys.path in
# the flows and the parity driver, exactly as runner.py's siblings do).
from config import EvalConfig
from runner import (
    _resolve_dataset,
    _resolve_num_situations,
    build_situations_prompt,
)
from risk_averse_prompts import default_system_prompt_for_dataset, resolve_system_prompt
from situations import build_situations, filter_lin_only_situations, validate_dataset_columns

from .scoring import summarize_results
from .dataset_schema_utils import ensure_option_level_dataframe

NAMESPACE = "riskaverse"  # openai-api service prefix (aligne uses "aligne")

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


ROW_METRICS = [
    parse_rate(), cooperate_rate(), rebel_rate(), steal_rate(),
    best_cara_rate(), best_linear_rate(), worst_linear_rate(),
    avg_expected_value_fraction_of_best(), avg_expected_value_relative_to_range(),
    avg_expected_value_regret(), num_total(), num_parsed(), num_parse_failed(),
]


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
