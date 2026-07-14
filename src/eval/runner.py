"""Library-first entrypoint for the generative-policy evaluation.

``run_evaluation(cfg, client)`` is the primary API: given a typed ``EvalConfig``
and an injected ``ChatClient`` (duck-typed OpenAI interface), it loads the
dataset, fans the situations out through the client, parses + scores the
responses, writes the per-run JSON, and returns an ``EvalResult``. There is no
``base_url`` anywhere in this surface — concurrency and caching come from the
client; the flow and the CLI decide which client to build.

It composes the split library — ``situations`` (loading), ``generation`` (the
client path), ``scoring`` (metrics), ``answer_parser`` (parsing) — and reuses
the parsing/scoring code paths verbatim, so metrics do not drift from the
original monolith. The ``vllm`` / ``transformers`` backends stay local (parity
anchor); with no injected client they are delegated to the CLI machinery.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from answer_parser import parse_choice_with_strategy
from config import EvalConfig, EvalResult
from dataset_schema_utils import ensure_option_level_dataframe
from generation import generate_openai
from risk_averse_prompts import default_system_prompt_for_dataset, resolve_system_prompt
from scoring import (
    project_failed_response_for_output,
    project_result_row_for_output,
    summarize_result_payload,
    summarize_results_by_field,
)
from situations import (
    DEFAULT_NUM_SITUATIONS_BY_DATASET,
    PROBABILITY_FORMATS,
    SOURCE_STAKES,
    SUBSET_TYPES,
    build_eval_prompt,
    build_situation_manifest,
    build_situations,
    filter_lin_only_situations,
    label_to_option_number,
    resolve_builtin_dataset_path,
    resolve_path,
    validate_dataset_columns,
)


def _resolve_dataset(cfg: EvalConfig):
    """Return (csv_path, resolved_variant, base_alias) for the config."""
    if cfg.custom_csv:
        path = resolve_path(cfg.custom_csv)
        return path, "custom", "custom"
    return resolve_builtin_dataset_path(cfg.dataset, cfg.dataset_variant)


def _resolve_num_situations(cfg: EvalConfig, df: pd.DataFrame) -> int:
    if cfg.num_situations is not None:
        return cfg.num_situations
    recommended = DEFAULT_NUM_SITUATIONS_BY_DATASET.get(cfg.dataset)
    if recommended is not None:
        return recommended
    return int(df["situation_id"].nunique())


def _build_result_row(sit: dict, eval_prompt: str, gen: dict) -> dict:
    """Parse + score one response into a result row.

    Moved verbatim from the monolith's core loop (``run_single_alpha_eval``):
    the parsing strategy, the option/EV bookkeeping, and the None-fallbacks are
    the measurement contract and must not drift.
    """
    response = gen["text"]
    num_generated_tokens = gen["num_tokens"]
    parse_result = parse_choice_with_strategy(
        response,
        sit["num_options"],
        label_style=sit.get("answer_label_style"),
        finish_reason=gen.get("finish_reason"),
    )
    choice = parse_result.choice
    parser_strategy = parse_result.strategy
    choice_index = label_to_option_number(choice) if choice else None

    result_row = {
        "situation_id": sit["situation_id"],
        "dataset_position": sit["dataset_position"],
        "subset_type": sit["subset_type"],
        "source_stakes": sit.get("source_stakes"),
        "source_condition": sit.get("source_condition"),
        "source_csv_name": sit.get("source_csv_name"),
        "source_situation_id": sit.get("source_situation_id"),
        "option_types_besides_cooperate": sit["option_types_besides_cooperate"],
        "prompt": eval_prompt,
        "num_options": sit["num_options"],
        "probability_format": sit["probability_format"],
        "bucket_label": sit["bucket_label"],
        "choice": choice if choice and choice in sit["options"] else None,
        "choice_index": choice_index if choice and choice in sit["options"] else None,
        "parser_strategy": parser_strategy,
        "response": response,
        "response_length": len(response),
        "num_tokens_generated": int(num_generated_tokens),
        "generation_finish_reason": gen.get("finish_reason"),
        "generation_stop_reason": gen.get("stop_reason"),
    }

    if choice and choice in sit["options"]:
        chosen = sit["options"][choice]
        result_row.update(
            {
                "option_type": chosen["type"],
                "is_best_cara": chosen["is_best_cara"],
                "is_best_linear": chosen["is_best_linear"],
                "is_worst_linear": chosen.get("is_worst_linear"),
                "expected_value": chosen.get("expected_value"),
                "max_expected_value": sit.get("max_expected_value"),
                "min_expected_value": sit.get("min_expected_value"),
                "expected_value_fraction_of_best": (
                    (chosen.get("expected_value") / sit.get("max_expected_value"))
                    if chosen.get("expected_value") is not None
                    and sit.get("max_expected_value") not in (None, 0)
                    else None
                ),
                "expected_value_relative_to_range": (
                    1.0
                    if chosen.get("expected_value") is not None
                    and sit.get("max_expected_value") is not None
                    and sit.get("min_expected_value") is not None
                    and abs(sit.get("max_expected_value") - sit.get("min_expected_value")) < 1e-12
                    else (
                        (chosen.get("expected_value") - sit.get("min_expected_value"))
                        / (sit.get("max_expected_value") - sit.get("min_expected_value"))
                    )
                    if chosen.get("expected_value") is not None
                    and sit.get("max_expected_value") is not None
                    and sit.get("min_expected_value") is not None
                    else None
                ),
                "expected_value_regret": (
                    sit.get("max_expected_value") - chosen.get("expected_value")
                    if chosen.get("expected_value") is not None
                    and sit.get("max_expected_value") is not None
                    else None
                ),
            }
        )
    else:
        result_row.update(
            {
                "option_type": None,
                "is_best_cara": None,
                "is_best_linear": None,
                "is_worst_linear": None,
                "expected_value": None,
                "max_expected_value": sit.get("max_expected_value"),
                "min_expected_value": sit.get("min_expected_value"),
                "expected_value_fraction_of_best": None,
                "expected_value_relative_to_range": None,
                "expected_value_regret": None,
            }
        )
    return result_row


def _write_output(cfg: EvalConfig, results: List[dict], summary: dict) -> Optional[str]:
    if not cfg.output:
        return None
    manifest = build_situation_manifest(results)
    failed = [r for r in results if r["option_type"] is None]
    payload = {
        "evaluation_config": {
            "backend": cfg.backend,
            "base_model": cfg.base_model,
            "dataset": cfg.dataset,
            "num_situations": len(results),
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "seed": cfg.seed,
            "max_new_tokens": cfg.max_new_tokens,
            "reasoning": {"max_tokens": cfg.reasoning_max_tokens},
            "system_prompt": cfg.system_prompt,
            "prompt_suffix": cfg.prompt_suffix,
        },
        "metrics": summary["metrics"],
        "num_valid": summary["num_valid"],
        "num_behaviorally_classified": summary["num_behaviorally_classified"],
        "num_total": summary["num_total"],
        "num_parse_failed": summary["num_parse_failed"],
        "metrics_by_subset_type": summarize_results_by_field(
            results, manifest, field_name="subset_type", ordered_values=list(SUBSET_TYPES)
        ),
        "metrics_by_source_stakes": summarize_results_by_field(
            results, manifest, field_name="source_stakes", ordered_values=list(SOURCE_STAKES)
        ),
        "results": [
            project_result_row_for_output(r, include_response=cfg.save_responses)
            for r in results
        ],
        "failed_responses_sample": [project_failed_response_for_output(r) for r in failed[-10:]],
    }
    out = Path(cfg.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: getattr(o, "item", lambda: str(o))())
    os.replace(tmp, out)
    return str(out)


async def run_evaluation(cfg: EvalConfig, client=None) -> EvalResult:
    """Run one generative-policy evaluation and return an ``EvalResult``.

    With an injected ``client`` (any object exposing ``await chat(payload)``)
    the situations are generated through it — the primary path. Without a client
    the ``vllm`` / ``transformers`` backends are delegated to the CLI machinery
    (imported lazily so this module stays torch-free); ``openai`` without a
    client is an error, since there is nothing to talk to.
    """
    if client is None:
        if cfg.backend in ("vllm", "transformers"):
            return await asyncio.to_thread(_run_local, cfg)
        raise ValueError(
            "run_evaluation with backend 'openai' needs an injected client "
            "(build one with serving.client(...))."
        )

    csv_path, resolved_variant, base_alias = _resolve_dataset(cfg)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

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
    # Record the RESOLVED prompt in the output's evaluation_config — the saved
    # run must show what the model actually saw, not the pre-resolution knob.
    cfg.system_prompt = system_prompt

    eval_prompts = [build_situations_prompt(sit, cfg) for sit in situations]
    gens = await generate_openai(
        client,
        eval_prompts=eval_prompts,
        system_prompt=system_prompt,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        seed=cfg.seed,
        max_new_tokens=cfg.max_new_tokens,
    )

    results = [
        _build_result_row(sit, prompt, gen)
        for sit, prompt, gen in zip(situations, eval_prompts, gens)
    ]
    summary = summarize_result_payload(results)
    output_path = _write_output(cfg, results, summary)

    return EvalResult(
        dataset=cfg.dataset,
        metrics=summary["metrics"],
        num_total=summary["num_total"],
        num_valid=summary["num_valid"],
        num_parse_failed=summary["num_parse_failed"],
        num_behaviorally_classified=summary["num_behaviorally_classified"],
        output_path=output_path,
        results=results,
    )


def build_situations_prompt(sit: dict, cfg: EvalConfig) -> str:
    return build_eval_prompt(sit["prompt_raw"], cfg.prompt_suffix)


def _run_local(cfg: EvalConfig):
    """Delegate a local-model (vllm/transformers) run to the CLI machinery.

    Imported lazily: ``evaluate`` pulls in torch, which the client path must not
    require.
    """
    import evaluate

    parser = evaluate.build_parser()
    args = parser.parse_args([])
    overrides = {
        "backend": cfg.backend,
        "base_model": cfg.base_model,
        "dataset": cfg.dataset,
        "dataset_variant": cfg.dataset_variant,
        "custom_csv": cfg.custom_csv,
        "num_situations": cfg.num_situations,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "seed": cfg.seed,
        "max_new_tokens": cfg.max_new_tokens,
        "reasoning_max_tokens": cfg.reasoning_max_tokens,
        "system_prompt": cfg.system_prompt,
        "prompt_suffix": cfg.prompt_suffix,
        "lin_only": cfg.lin_only,
        "start_position": cfg.start_position,
        "end_position": cfg.end_position,
        "output": cfg.output,
        "allow_nondefault_temperature": True,
    }
    for k, v in overrides.items():
        setattr(args, k, v)
    summary = evaluate.run_cli_evaluation(args)
    metrics = summary.get("metrics") or {}
    return EvalResult(
        dataset=cfg.dataset,
        metrics=metrics,
        num_total=summary.get("num_total") or 0,
        num_valid=summary.get("num_valid") or 0,
        num_parse_failed=summary.get("num_parse_failed") or 0,
        num_behaviorally_classified=summary.get("num_behaviorally_classified") or 0,
        output_path=summary.get("output_path"),
    )
