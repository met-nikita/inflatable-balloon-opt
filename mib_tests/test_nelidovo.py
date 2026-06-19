"""Reproduce the paper's worked example: mu* = 1.08."""

from __future__ import annotations

from balloon import BalloonSolver
from balloon.exact import solve_exact
from examples.nelidovo_1977 import build_problem


def test_nelidovo_optimum_is_1_08():
    p = build_problem()
    res = BalloonSolver(p, eps=1e-5).solve()
    # the paper reports mu = 1.08 to two decimals
    assert abs(res.mu - 1.08) < 0.01
    assert res.allocation.satisfies()


def test_nelidovo_matches_exact_lp():
    p = build_problem()
    res = BalloonSolver(p, eps=1e-5).solve()
    ex = solve_exact(p)
    assert abs(res.mu - ex.mu) < 1e-3


def test_problem_shape():
    p = build_problem()
    assert p.m == 20
    assert p.n == 16
    # product 10 has the lower productivity 1.44
    assert abs(p.a[2, 9] - 1.44) < 1e-12  # aggregate 3 -> product 10


if __name__ == "__main__":
    from mib_tests.run_all import run_module

    run_module(__name__)
