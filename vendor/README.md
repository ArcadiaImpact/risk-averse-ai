# vendor/

## riskaverseAIs

The [riskaverseAIs benchmark](https://github.com/riskaverseAIs/riskaverseAIs)
(anonymous reproduction release for Thornley & MacAskill, *Risk-Averse AIs*,
Forethought 2026), committed verbatim at upstream commit
`79f2da1a838db00d5704aeaecd4d6b3fd1110967` (imported 2026-07-14, `.git`
stripped).

- **Licenses:** upstream code is MIT (`riskaverseAIs/LICENSE`); the
  evaluation datasets are CC-BY-4.0
  (`riskaverseAIs/evaluation/LICENSE-CC-BY-4.0.txt`). Both are preserved
  in-tree.
- **Local modifications are allowed** and tracked by git — the import commit
  is the pristine baseline, so `git log -- vendor/riskaverseAIs` (or a diff
  against the import commit) shows exactly how we diverge from upstream.
  Keep divergence minimal and deliberate.
- The pin recorded in `config.yaml` (`benchmark.commit`) documents which
  upstream commit this snapshot corresponds to; it is provenance metadata
  now, not a fetch instruction (the old `scripts/fetch_benchmark.sh` is
  retired).
- **The benchmark is held out**: never train on its gamble format (see
  `CLAUDE.md`).
