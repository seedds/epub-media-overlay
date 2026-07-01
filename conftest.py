"""Make the repo root importable for tests.

The test suite lives in `tests/` but imports the source modules by their bare
names (`import pipeline_core`, `from mark_sentence import ...`). Under pytest the
`pythonpath = ["."]` setting in pyproject.toml handles this. This conftest adds
the same path so the tests' no-pytest `__main__` self-runners also work when a
file is executed directly (e.g. `python tests/test_segmentation.py`).
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
