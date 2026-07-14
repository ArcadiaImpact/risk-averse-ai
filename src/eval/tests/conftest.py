"""Put the eval package dir on sys.path so tests can import the modules under
test by their bare names (``from answer_parser import ...``), matching how the
evaluation modules import their siblings at runtime."""
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
