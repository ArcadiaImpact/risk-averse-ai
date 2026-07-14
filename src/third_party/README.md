# src/third_party/

## riskaverseAIs

The [riskaverseAIs benchmark](https://github.com/riskaverseAIs/riskaverseAIs)
(anonymous reproduction release for Thornley & MacAskill, *Risk-Averse AIs*,
Forethought 2026), committed verbatim at upstream commit
`79f2da1a838db00d5704aeaecd4d6b3fd1110967` (imported 2026-07-14, `.git`
stripped).

This tree holds the upstream **training-method arms only** — `sft-training`,
`dpo-training`, `steering`, `tie-training`, `reward-model`,
`dataset-generation`, plus the top-level `LICENSE` and `README`. It is
**reference-only**: we don't import or run it.

- **The `evaluation/` subtree was lifted out to `src/eval/`** and is now
  first-party-maintained (an endpoint/server backend is planned). It started
  as a verbatim copy of upstream `evaluation/` @ `79f2da1` and is allowed to
  diverge; its licenses (MIT + CC-BY-4.0 for the eval data) travelled with it.
  See `src/eval/README.md`.
- **Licenses:** upstream code is MIT (`riskaverseAIs/LICENSE`), preserved
  in-tree. The evaluation-data licenses moved with the eval tree to
  `src/eval/` (`LICENSE`, `LICENSE-CC-BY-4.0.txt`, `DATA_LICENSE.md`).
- **Local modifications are allowed** and tracked by git — the import commit
  is the pristine baseline, so `git log -- src/third_party/riskaverseAIs`
  (or a diff against the import commit) shows exactly how we diverge from
  upstream. Keep divergence minimal and deliberate.
- The pin recorded in `configs/config.yaml` (`benchmark.commit`) documents
  which upstream commit this snapshot corresponds to; it is provenance
  metadata now, not a fetch instruction (the old `scripts/fetch_benchmark.sh`
  is retired).
- **The benchmark is held out**: never train on its gamble format (see
  `CLAUDE.md`).
