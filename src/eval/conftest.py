"""Put the eval dir (and repo ``src/``) on sys.path for every test under
``src/eval`` — the per-task tests, the utils tests, and the legacy-path tests
all import the modules under test by the same bare/``utils.``-prefixed names the
runtime uses (``from config import ...``, ``from utils.answer_parser import
...``, ``import tasks``). One conftest at the eval root serves them all.
"""
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent          # src/eval
SRC_DIR = EVAL_DIR.parent                            # src
for _p in (EVAL_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
