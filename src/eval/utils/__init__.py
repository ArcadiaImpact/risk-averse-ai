"""Shared eval helpers, defined once and reused across every task.

This package is the home for code that is genuinely common to more than one
task (or to both the legacy runner and the inspect battery), so a
``tasks/<name>/`` dir can hold only what is peculiar to that task:

* :mod:`~utils.answer_parser` — the benchmark's permissive choice parser;
* :mod:`~utils.scoring` — ``summarize_results`` and the metric/row shapes;
* :mod:`~utils.dataset_schema_utils` — option-level dataframe helpers;
* :mod:`~utils.cara` / :mod:`~utils.lotteries` — the CARA/lottery math shared by
  the OOD generators and scorers;
* :mod:`~utils.ood_schema` / :mod:`~utils.ood_fmt` / :mod:`~utils.ood_common` —
  the OOD item schema and generator machinery;
* :mod:`~utils.ood_scoring` — the shared pick-one OOD scorer + tool-call adapter;
* :mod:`~utils.inspect_shared` — the cross-task inspect_ai glue (model seam,
  metrics, sample builders, EvalLog adapters).

Modules are imported by submodule path (``from utils.answer_parser import ...``),
resolving because ``src/eval`` is on ``sys.path`` in the flows, the parity
driver, and the tests — exactly as the legacy modules import their siblings by
bare name. Importing :mod:`utils.inspect_shared` pulls inspect_ai, so it (like
the tasks package) is imported explicitly by the inspect backend, never at
package import time.
"""
