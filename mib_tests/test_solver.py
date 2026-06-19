"""Balloon inflation: exact on sparse problems, a valid lower bound elsewhere."""

from __future__ import annotations

import numpy as np

from balloon import BalloonSolver, Problem, solve
from balloon.exact import LPOracle, solve_exact
from mib_tests.helpers import random_problem


def test_balloon_exact_on_sparse_problems():
    """On sparse matrices the class-fixing method reaches the LP optimum."""
    rng = np.random.default_rng(7)
    for _ in range(80):
        p = random_problem(rng, density=0.3)
        res = BalloonSolver(p, eps=1e-5).solve()
        ex = solve_exact(p)
        assert abs(res.mu - ex.mu) <= 1e-3 * (1 + ex.mu), (
            f"balloon mu={res.mu} vs exact mu={ex.mu}"
        )
        assert res.allocation.satisfies()


def test_balloon_is_always_a_valid_lower_bound():
    """Even on dense matrices the method never *over*-reports the optimum."""
    rng = np.random.default_rng(99)
    for _ in range(60):
        p = random_problem(rng, density=0.8)
        res = BalloonSolver(p, eps=1e-5).solve()
        ex = solve_exact(p)
        assert res.mu <= ex.mu + 1e-6, "reported mu exceeds the true optimum"
        assert res.allocation.satisfies(), "returned allocation must be feasible"


def test_lp_oracle_makes_the_search_exact_anywhere():
    """Plugging in the exact oracle recovers the LP optimum on dense problems."""
    rng = np.random.default_rng(100)
    for _ in range(30):
        p = random_problem(rng, density=0.9)
        res = BalloonSolver(p, eps=1e-5, oracle=LPOracle(p)).solve()
        ex = solve_exact(p)
        assert abs(res.mu - ex.mu) <= 1e-3 * (1 + ex.mu)


def test_known_2x2():
    p = Problem.from_lists(a=[[2.0, 1.0], [1.0, 2.0]], z=[1.0, 1.0], b0=[1.0, 1.0])
    res = solve(p, eps=1e-7)
    assert abs(res.mu - 2.0) < 1e-4


def test_history_is_recorded():
    p = Problem.from_lists(a=[[2.0, 1.0], [1.0, 2.0]], z=[1.0, 1.0], b0=[1.0, 1.0])
    res = BalloonSolver(p, eps=1e-4).solve()
    assert len(res.history) > 0
    assert all(h.kind in ("grow", "dichotomy", "probe", "bisect") for h in res.history)
    assert res.history[0].mu < res.mu + 1e-9
    # the search must visit both feasible and infeasible mu (it brackets mu*)
    assert any(not h.feasible for h in res.history)


def test_single_bottleneck():
    # One enterprise, one product: mu* = a*z / b0.
    p = Problem.from_lists(a=[[3.0]], z=[4.0], b0=[2.0])
    res = solve(p, eps=1e-7)
    assert abs(res.mu - (3.0 * 4.0 / 2.0)) < 1e-4  # = 6.0


if __name__ == "__main__":
    from mib_tests.run_all import run_module

    run_module(__name__)
