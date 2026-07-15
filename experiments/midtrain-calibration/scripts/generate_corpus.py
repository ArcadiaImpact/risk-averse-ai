"""Stage (i): expand behavior_spec.md into a synthetic-document midtraining corpus.

The model-spec-midtraining generation pattern (cf. science-of-midtraining
``src/scimt/gen``), but WRAPPING ``aligne.synthdoc`` directly — we do not import
scimt. Each of ``n_batches`` independent ``generate_corpus`` calls re-plans
``n_domains`` domains at temperature and emits ``docs_per_domain`` docs/domain;
the corpora are concatenated (no cross-batch dedup) so the union spans far more
settings than one plan. The seed is the prose behavioral profile in
``behavior_spec.md`` — never a benchmark-format lottery menu.

Writes to ``<out_dir>/``:
  - ``corpus.jsonl``   — one ``{"text": ..., <docspec meta>}`` per line
  - ``dataset.jsonl``  — one ``{"messages": [{"role":"assistant","content":text}]}``
                         per line, ready for the aligne SFT driver (a lone
                         assistant turn = continued-pretraining / doc finetune)
  - ``gen_manifest.json`` — provenance, stats, and the leakage AUDIT result
  - ``sample_docs.jsonl`` — ~20 docs, committed to git (full corpus is gitignored)

Held-out audit: after generation every doc is scanned for the benchmark's
gamble-menu format markers (BENCHMARK_LEAK_PATTERNS). Any hit is reported per
doc in the manifest; a clean corpus reports ``leak_docs: 0``.

    uv run python experiments/midtrain-calibration/scripts/generate_corpus.py \
        --config configs/gen.yaml
    # audit an already-generated corpus without regenerating:
    uv run python experiments/midtrain-calibration/scripts/generate_corpus.py \
        --config configs/gen.yaml --audit-only
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

# --- held-out audit: the benchmark's gamble-menu format markers -------------
# The riskaverseAIs benchmark presents two-option lottery menus like
#   "a. probability 0.82 of $749, probability 0.18 ..." inside a prompt that
# says "You are turned into an artificial agent ... select whichever option
# seems most attractive". None of that may appear in a midtraining doc — the
# gamble FORMAT is held out from every arm. Worked prose reasoning about
# decisions (present in the seed) is fine; the benchmark's task framing is not.
BENCHMARK_LEAK_PATTERNS = {
    "option_prob_menu": re.compile(r"probability\s+0?\.\d+\s+of\s+\$", re.I),
    "gamble_choice_tasks": re.compile(r"gamble[- ]choice task", re.I),
    "select_whichever_option": re.compile(r"select whichever option", re.I),
    "turned_into_agent": re.compile(r"turned into an artificial agent", re.I),
    "lettered_option_menu": re.compile(r"(?m)^\s*[a-e][.)]\s+probability", re.I),
    "current_bank_balance_menu": re.compile(r"you have determined that you have the following options", re.I),
}


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_env(path: Path = Path.home() / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def audit_records(records: list[dict]) -> dict:
    """Scan every doc for benchmark-format leakage. Returns an audit dict."""
    hits: dict[str, int] = {name: 0 for name in BENCHMARK_LEAK_PATTERNS}
    leak_examples: list[dict] = []
    leak_docs = 0
    for i, rec in enumerate(records):
        text = str(rec.get("text", ""))
        matched = [name for name, pat in BENCHMARK_LEAK_PATTERNS.items() if pat.search(text)]
        if matched:
            leak_docs += 1
            for name in matched:
                hits[name] += 1
            if len(leak_examples) < 10:
                leak_examples.append({"doc_index": i, "markers": matched,
                                      "excerpt": text[:200]})
    return {
        "n_docs_scanned": len(records),
        "leak_docs": leak_docs,
        "leak_rate": (leak_docs / len(records)) if records else 0.0,
        "marker_hits": hits,
        "leak_examples": leak_examples,
        "patterns": {k: v.pattern for k, v in BENCHMARK_LEAK_PATTERNS.items()},
        "clean": leak_docs == 0,
    }


async def _generate(cfg: dict) -> list[dict]:
    from aligne.client import ChatClient, Endpoint
    from aligne.synthdoc import Spec, SynthdocConfig, generate_corpus

    seed_text = (REPO_ROOT / cfg["seed"]).read_text()
    spec = Spec(
        name=cfg["spec_name"],
        text=seed_text,
        assistant_name=cfg.get("assistant_name", "the assistant"),
        provider_name=cfg.get("provider_name", "the lab"),
    )
    sd_cfg = SynthdocConfig(
        n_domains=cfg["n_domains"],
        docs_per_domain=cfg["docs_per_domain"],
        target_words=cfg["target_words"],
        critique=cfg.get("critique", True),
        dedup_threshold=cfg.get("dedup_threshold", 0.7),
        temperature=cfg.get("temperature", 1.0),
        planner_chunk_size=cfg.get("planner_chunk_size", 4),
        plan_retries=cfg.get("plan_retries", 3),
        on_domain_failure=cfg.get("on_domain_failure", "drop"),
    )

    key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))
    ep = Endpoint(cfg["base_url"], cfg["model"], api_key=key)
    client = ChatClient(ep, concurrency=cfg.get("concurrency", 32))
    records: list[dict] = []
    n_batches = max(1, cfg.get("n_batches", 1))
    try:
        for b in range(n_batches):
            result = await generate_corpus(client, spec, sd_cfg)
            for doc in result.documents:
                meta = dataclasses.asdict(doc.spec)
                meta["tokens_est"] = doc.tokens_est
                meta["batch"] = b
                rec = {"text": doc.text}
                rec.update({k: v for k, v in meta.items() if v is not None})
                records.append(rec)
            print(f"[gen] batch {b+1}/{n_batches}: +{len(result.documents)} docs "
                  f"(dropped {len(result.dropped)}, failed domains "
                  f"{len(result.failed_domains)}); total {len(records)}",
                  flush=True)
    finally:
        await client.aclose()
    return records


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_corpus(records: list[dict], out_dir: Path, cfg: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    dataset_path = out_dir / "dataset.jsonl"
    sample_path = out_dir / "sample_docs.jsonl"
    _write_jsonl(corpus_path, records)
    _write_jsonl(dataset_path,
                 [{"messages": [{"role": "assistant", "content": r["text"]}]} for r in records])
    # Committed sample: ~20 docs, spread across batches for variety.
    step = max(1, len(records) // 20)
    _write_jsonl(sample_path, records[::step][:20])

    audit = audit_records(records)
    manifest = {
        "spec": cfg["spec_name"],
        "seed": cfg["seed"],
        "source": "synthdoc",
        "gen_model": cfg["model"],
        "n_batches": cfg.get("n_batches", 1),
        "n_domains": cfg["n_domains"],
        "docs_per_domain": cfg["docs_per_domain"],
        "target_words": cfg["target_words"],
        "seed_value": cfg.get("seed_value"),
        "n_docs": len(records),
        "corpus_path": str(corpus_path.relative_to(REPO_ROOT)),
        "dataset_path": str(dataset_path.relative_to(REPO_ROOT)),
        "sample_path": str(sample_path.relative_to(REPO_ROOT)),
        "audit": audit,
    }
    (out_dir / "gen_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/gen.yaml")
    ap.add_argument("--audit-only", action="store_true",
                    help="re-audit an existing corpus.jsonl without regenerating")
    args = ap.parse_args()

    cfg = load_cfg(EXP_DIR / args.config)
    load_env()
    out_dir = REPO_ROOT / cfg["out_dir"]

    if args.audit_only:
        records = [json.loads(l) for l in (out_dir / "corpus.jsonl").read_text().splitlines() if l.strip()]
        audit = audit_records(records)
        print(json.dumps(audit, indent=2))
        return

    records = asyncio.run(_generate(cfg))
    manifest = write_corpus(records, out_dir, cfg)
    print(f"[gen] wrote {manifest['n_docs']} docs -> {out_dir}")
    a = manifest["audit"]
    print(f"[audit] leak_docs={a['leak_docs']}/{a['n_docs_scanned']} "
          f"clean={a['clean']} marker_hits={a['marker_hits']}")
    if not a["clean"]:
        print("[audit] WARNING: benchmark-format leakage detected; see gen_manifest.json", file=sys.stderr)


if __name__ == "__main__":
    main()
