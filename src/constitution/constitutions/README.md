# Vendored constitutions

These three constitution JSONs are vendored **byte-for-byte** from
[ArcadiaImpact/aligne](https://github.com/ArcadiaImpact/aligne)
@ `18bd0798` (`src/aligne/character/constitutions/`):

- `risk_averse.json`
- `risk_averse_calibrated.json`
- `risk_seeking.json`

The canonical home is aligne. Do **not** edit them here except to re-vendor
from aligne — `constitution.py` (also vendored from the same commit) renders
them, and `scripts/render_parity.py` checks byte-parity of the render against a
live aligne checkout. The texts are already public in Appendix A of
`reports/2026-07-10-distill-v1.md`.
