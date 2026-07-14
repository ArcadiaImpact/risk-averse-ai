# Vendored from ArcadiaImpact/aligne @ a907ac83 (PR #12)
# (src/aligne/train/tinker/cli.py). Canonical home is aligne; this is a
# byte-for-byte copy of the parts we keep. Do NOT edit here except to re-vendor
# from aligne.
#
# STRIPPED on vendor: the argparse scaffolding (``out_explicit``,
# ``add_common_tinker_args``, ``apply_smoke``) — we call the typed function API
# (``distill_reverse_kl``) directly and never build a parser, so only the
# default-renderer constant is retained.

"""Shared constant for the Tinker training drivers (renderer default)."""

from __future__ import annotations

# The non-thinking renderer the EM experiment trained/eval'd with. Qwen3.6
# family renders as qwen3_5; non-thinking matches the plain bad-medical data.
DEFAULT_RENDERER = "qwen3_5_disable_thinking"
