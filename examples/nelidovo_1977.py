"""The worked numerical example from Bronfeld's 1977 paper (Tables 1 & 2).

A plan for an assortment of **16 products** on **20 aggregates** (machines).
Table 1 lists the demand constraints (2): for each product *j*, the aggregates
that can make it (all with productivity 1.52, except product 10 at 1.44) and the
assortment requirement ``b0[j]`` (the coefficient of ``mu``).  Table 2 lists the
time resources ``z[i]`` of each aggregate (the right-hand sides of (3)).

The paper reports ``mu* = 1.08`` (to 0.01), found by slide rule in 6 iterations
with 16 estimate-fixing steps.  Run this module to reproduce it with the
implementation and cross-check it against the exact LP optimum.

    python -m examples.nelidovo_1977
"""

from __future__ import annotations

import numpy as np

from balloon import BalloonSolver, GeneralizedBalloonSolver, Problem

# --- enterprises (aggregates) are numbered 1..20, products 1..16 in the paper.
M, N = 20, 16

# Table 1, read as (enterprise, product) -> productivity a'_ij.
# Productivity is 1.52 everywhere except product 10 (j=10) which is 1.44.
# (i, j) pairs taken verbatim from the demand rows of Table 1.
_TABLE1_PAIRS = [
    # product 1
    (7, 1), (8, 1),
    # product 2
    (1, 2), (2, 2), (19, 2),
    # product 3
    (8, 3),
    # product 4
    (7, 4),
    # product 5
    (7, 5),
    # product 6
    (18, 6), (20, 6),
    # product 7
    (3, 7), (4, 7), (18, 7), (20, 7),
    # product 8
    (3, 8), (4, 8), (18, 8),
    # product 9
    (17, 9),
    # product 10  (productivity 1.44)
    (3, 10), (4, 10), (13, 10), (14, 10), (18, 10),
    # product 11
    (3, 11), (4, 11), (5, 11), (18, 11),
    # product 12
    (3, 12), (4, 12), (5, 12), (18, 12),
    # product 13
    (3, 13), (4, 13), (5, 13), (17, 13),
    # product 14
    (3, 14), (4, 14), (6, 14), (9, 14), (10, 14), (15, 14), (16, 14),
    # product 15
    (11, 15), (12, 15), (18, 15),
    # product 16
    (11, 16), (12, 16), (18, 16),
]

# Assortment requirements b0[j] (coefficient of mu in each demand row of Table 1).
_B0 = {
    1: 51, 2: 216, 3: 69, 4: 33, 5: 72, 6: 51, 7: 60, 8: 42,
    9: 60, 10: 114, 11: 30, 12: 90, 13: 33, 14: 318, 15: 24, 16: 111,
}

# Time resources z[i] per aggregate (right-hand sides of Table 2).
_Z = {
    1: 16, 2: 60, 3: 16, 4: 60, 5: 80, 6: 80, 7: 80, 8: 80, 9: 37, 10: 39,
    11: 37, 12: 39, 13: 40, 14: 18, 15: 40, 16: 18, 17: 80, 18: 50, 19: 80, 20: 80,
}


def build_problem() -> Problem:
    """Assemble the 20x16 :class:`~balloon.problem.Problem` from the paper."""
    a = np.zeros((M, N))
    for i, j in _TABLE1_PAIRS:
        a[i - 1, j - 1] = 1.44 if j == 10 else 1.52
    z = np.array([_Z[i + 1] for i in range(M)], dtype=float)
    b0 = np.array([_B0[j + 1] for j in range(N)], dtype=float)
    enterprises = tuple(f"A{i+1:02d}" for i in range(M))
    products = tuple(f"P{j+1:02d}" for j in range(N))
    return Problem(a=a, z=z, b0=b0, enterprises=enterprises, products=products)


def main() -> None:
    p = build_problem()
    print(f"Nelidovo 1977 example: {p.m} aggregates x {p.n} products")
    print("-" * 60)

    res = BalloonSolver(p, eps=1e-5).solve()
    print(f"Balloon Method  mu* = {res.mu:.4f}   "
          f"(paper reports 1.08; boundary estimate mu_gr = {res.mu_gr:.3f})")
    print(f"  inflation steps: {res.iterations}, allocation feasible: "
          f"{res.allocation.satisfies()}")

    try:
        from balloon.exact import solve_exact

        ex = solve_exact(p)
        print(f"Exact LP        mu* = {ex.mu:.4f}")
        print(f"  match: {abs(ex.mu - res.mu) < 1e-3}")
    except Exception as exc:  # pragma: no cover - scipy optional at runtime
        print(f"(exact LP check skipped: {exc})")

    print("\nTime allocation x[aggregate, product] (non-zero entries):")
    for (e, prod), t in sorted(res.allocation.nonzero(tol=1e-4).items()):
        print(f"    {e} -> {prod}: {t:8.3f}")

    print("\nResource utilisation per aggregate (used / available):")
    used, z = res.allocation.used_time, p.z
    for i in range(p.m):
        if z[i] > 0:
            print(f"    {p.enterprises[i]}: {used[i]:7.3f} / {z[i]:6.1f}"
                  f"  ({100*used[i]/z[i]:5.1f}%)")

    # Generalised balloon: use leftover capacity to inflate individual criteria.
    gen = GeneralizedBalloonSolver(p, eps=1e-4).solve()
    extra = gen.demand / p.b0  # achievable per-product "mu" after bulging
    print(f"\nGeneralised balloon: uniform mu* = {gen.mu:.4f}, "
          f"then per-criterion inflation (rounds={gen.rounds}).")
    print("  per-product achievable sets b[j]/b0[j] after bulging into free space:")
    print("   ", np.array2string(extra, precision=2, max_line_width=100))


if __name__ == "__main__":
    main()
