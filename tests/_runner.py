"""Shared no-pytest fallback runner for the test files.

Each `tests/test_*.py` is a normal pytest module, but also runs standalone via
its `__main__` block so the suite works even without pytest installed. That block
just calls `run_module_tests(globals())` here instead of copy-pasting the loop.

Running a test file directly (`python tests/test_x.py`) does not trigger pytest's
conftest, so this module also puts the repo root on sys.path, mirroring the root
conftest.py, so the bare `import pipeline_core` style imports resolve.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def run_module_tests(namespace: dict) -> int:
    """Run every `test_*` callable in `namespace`; print PASS/FAIL/ERROR per test.

    Returns a process exit code (0 = all passed, 1 = any failure/error) so callers
    can `raise SystemExit(run_module_tests(globals()))`.
    """
    funcs = [
        (name, obj)
        for name, obj in sorted(namespace.items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in funcs:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append((name, exc))
            print(f"ERROR {name}: {exc!r}")
    print(f"\n{len(funcs) - len(failures)}/{len(funcs)} passed")
    return 1 if failures else 0
