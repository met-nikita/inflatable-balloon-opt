"""Generalised per-criterion inflation: feasibility, monotonicity, frontier."""

from __future__ import annotations

import numpy as np

from balloon import GeneralizedBalloonSolver, Problem, solve_generalized
from balloon.feasibility import FeasibilitySolver
from mib_tests.helpers import random_problem


def test_generalized_stays_feasible_and_monotone():
    rng = np.random.default_rng(11)
    for _ in range(40):
        p = random_problem(rng)
        gen = solve_generalized(p, eps=1e-4)
        assert gen.allocation.satisfies(), "generalised allocation must be feasible"
        # phase 2 never lowers a demand below the uniform level
        assert np.all(gen.demand >= gen.base_demand - 1e-6)


def test_caps_at_base_disable_phase_two():
    rng = np.random.default_rng(12)
    p = random_problem(rng)
    base = solve_generalized(p, eps=1e-4).base_demand
    gen = GeneralizedBalloonSolver(p, eps=1e-4, caps=base).solve()
    assert np.allclose(gen.demand, base, atol=1e-6)


def test_inflation_reaches_the_resource_frontier():
    """After inflation no single criterion can grow further on its own."""
    rng = np.random.default_rng(13)
    for _ in range(20):
        p = random_problem(rng)
        gen = solve_generalized(p, eps=1e-4)
        oracle = FeasibilitySolver(p)
        col_cap = (p.a * p.z[:, None]).sum(axis=0)
        for j in range(p.n):
            if gen.demand[j] >= col_cap[j] - 1e-6:
                continue  # already at the column's absolute ceiling
            bumped = gen.demand.copy()
            bumped[j] += 0.05 * (1 + gen.demand[j])
            assert not oracle.is_feasible(bumped), (
                f"product {j} could still be inflated -> not on the frontier"
            )


def test_respects_explicit_caps():
    rng = np.random.default_rng(14)
    p = random_problem(rng)
    caps = p.b0 * 100.0  # generous ceilings, but finite
    caps[0] = p.b0[0] * 0.0 + 1e-3  # force product 0 nearly to zero ceiling
    gen = GeneralizedBalloonSolver(p, eps=1e-4, caps=caps).solve()
    assert gen.demand[0] <= caps[0] + 1e-6


if __name__ == "__main__":
    from mib_tests.run_all import run_module

    run_module(__name__)
