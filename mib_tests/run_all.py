"""Minimal test runner (pytest is not required).

Run the whole suite from the project root with::

    python -m tests.run_all

or a single module with::

    python -m tests.test_solver
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
import types

TEST_MODULES = [
    "mib_tests.test_feasibility",
    "mib_tests.test_solver",
    "mib_tests.test_nelidovo",
    "mib_tests.test_nonlinear",
]


def _run_namespace(ns: dict, label: str) -> tuple[int, int]:
    funcs = [(n, f) for n, f in sorted(ns.items())
             if n.startswith("test_") and isinstance(f, types.FunctionType)]
    passed = failed = 0
    for name, fn in funcs:
        t0 = time.perf_counter()
        try:
            fn()
            dt = time.perf_counter() - t0
            print(f"  PASS  {label}.{name}  ({dt:.2f}s)")
            passed += 1
        except Exception:  # noqa: BLE001 - report any failure
            print(f"  FAIL  {label}.{name}")
            traceback.print_exc()
            failed += 1
    return passed, failed


def run_module(name: str) -> int:
    """Run all ``test_*`` functions in module ``name`` (or "__main__")."""
    if name == "__main__":
        mod = sys.modules["__main__"]
        label = getattr(mod, "__file__", "main")
    else:
        mod = importlib.import_module(name)
        label = name
    passed, failed = _run_namespace(vars(mod), label.rsplit(".", 1)[-1].replace(".py", ""))
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    total_pass = total_fail = 0
    for name in TEST_MODULES:
        print(f"\n=== {name} ===")
        mod = importlib.import_module(name)
        p, f = _run_namespace(vars(mod), name.rsplit(".", 1)[-1])
        total_pass += p
        total_fail += f
    print("\n" + "=" * 50)
    print(f"TOTAL: {total_pass} passed, {total_fail} failed")
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
