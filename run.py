#!/usr/bin/env python
"""Command-line driver for the Inflatable Balloon Method.

Read a problem from a JSON file (or stdin) and solve it.

Examples
--------
    python run.py examples/sample_problem.json
    python run.py examples/sample_problem.json --exact --trace
    python run.py examples/sample_problem.json --generalized

JSON schema::

    {
      "a":  [[2.0, 1.0], [1.0, 2.0]],   # m x n productivity matrix a'[i][j]
      "z":  [1.0, 1.0],                 # m   working-time resources
      "b0": [1.0, 1.0],                 # n   assortment requirements (> 0)
      "enterprises": ["E0", "E1"],      # optional labels
      "products":    ["P0", "P1"]       # optional labels
    }
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from balloon import (
    BalloonSolver,
    GeneralizedBalloonSolver,
    Problem,
)


def load_problem(path: str) -> Problem:
    text = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    data = json.loads(text)
    return Problem.from_lists(
        a=data["a"],
        z=data["z"],
        b0=data["b0"],
        enterprises=data.get("enterprises", ()),
        products=data.get("products", ()),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inflatable Balloon Method solver")
    ap.add_argument("problem", help="path to a JSON problem ('-' for stdin)")
    ap.add_argument("--eps", type=float, default=1e-5, help="convergence tolerance (eq. 9)")
    ap.add_argument("--exact", action="store_true",
                    help="use the exact LP oracle (guaranteed optimum, needs scipy)")
    ap.add_argument("--generalized", action="store_true",
                    help="also run per-criterion (nonlinear) inflation")
    ap.add_argument("--caps", type=str, default="",
                    help="comma-separated demand ceilings for --generalized")
    ap.add_argument("--trace", action="store_true", help="print the inflation trace")
    args = ap.parse_args(argv)

    p = load_problem(args.problem)
    oracle = None
    if args.exact:
        from balloon.exact import LPOracle

        oracle = LPOracle(p)

    res = BalloonSolver(p, eps=args.eps, oracle=oracle).solve()
    print(f"Problem: {p.m} enterprises x {p.n} products"
          f"   (oracle: {'exact LP' if args.exact else 'class-fixing'})")
    print(f"mu* = {res.mu:.6f}   (boundary estimate mu_gr = {res.mu_gr:.4f}, "
          f"{res.iterations} steps)")
    print(f"allocation feasible: {res.allocation.satisfies()}")

    if args.trace:
        print("\ninflation trace (t, mu, feasible, kind):")
        for h in res.history:
            print(f"  t={h.t:3d}  mu={h.mu:10.5f}  "
                  f"{'feasible' if h.feasible else 'INFEAS  '}  {h.kind}")

    print("\nnon-zero allocation x[enterprise, product]:")
    for (e, prod), t in sorted(res.allocation.nonzero(tol=1e-6).items()):
        print(f"  {e} -> {prod}: {t:.4f}")

    print("\nresource utilisation (used / available):")
    used = res.allocation.used_time
    for i in range(p.m):
        print(f"  {p.enterprises[i]}: {used[i]:.3f} / {p.z[i]:.3f}")

    if args.generalized:
        caps = None
        if args.caps:
            caps = np.array([float(c) for c in args.caps.split(",")], dtype=float)
        gen = GeneralizedBalloonSolver(p, eps=args.eps, caps=caps, oracle=oracle).solve()
        print(f"\nGeneralised balloon: uniform mu* = {gen.mu:.5f}, "
              f"phase-2 rounds = {gen.rounds}")
        print("  per-product demand after bulging  b[j]  (uniform -> inflated):")
        for j in range(p.n):
            print(f"    {p.products[j]}: {gen.base_demand[j]:.3f} -> {gen.demand[j]:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
