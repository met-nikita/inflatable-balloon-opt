"""The branch-and-bound feasibility oracle must agree with the exact LP oracle."""

from __future__ import annotations

import numpy as np

from balloon import Problem
from balloon.feasibility import FeasibilitySolver
from balloon.exact import feasible_exact
from mib_tests.helpers import max_scale, random_problem


def test_oracle_matches_lp_with_margin():
    """At 0.9x the LP-feasible scale demand must be feasible; at 1.1x infeasible."""
    rng = np.random.default_rng(0)
    for _ in range(60):
        p = random_problem(rng)
        oracle = FeasibilitySolver(p)
        direction = rng.uniform(1.0, 5.0, p.n)
        t = max_scale(p, direction)
        if t <= 1e-6:
            continue
        assert oracle.is_feasible(0.9 * t * direction), "should be feasible below the wall"
        assert not oracle.is_feasible(1.1 * t * direction), "should be infeasible above the wall"


def test_returned_allocation_actually_satisfies_constraints():
    rng = np.random.default_rng(1)
    for _ in range(40):
        p = random_problem(rng)
        oracle = FeasibilitySolver(p)
        direction = rng.uniform(1.0, 5.0, p.n)
        t = max_scale(p, direction)
        alloc = oracle.solve(0.8 * t * direction)
        assert alloc is not None
        assert alloc.satisfies(), "oracle returned an infeasible allocation"


def test_zero_demand_is_feasible():
    p = Problem.from_lists(a=[[1.0, 0.0], [0.0, 1.0]], z=[1.0, 1.0], b0=[1.0, 1.0])
    assert FeasibilitySolver(p).is_feasible([0.0, 0.0])


def test_unsuppliable_product_is_infeasible():
    # product 1 (column index 1) has no supplier -> any positive demand infeasible.
    p = Problem.from_lists(a=[[1.0, 0.0], [1.0, 0.0]], z=[1.0, 1.0], b0=[1.0, 1.0])
    oracle = FeasibilitySolver(p)
    assert oracle.is_feasible([1.0, 0.0])
    assert not oracle.is_feasible([1.0, 0.0001])


def test_forced_chain_a_and_b_classes():
    # A diagonal problem: every product has exactly one supplier (all A-class),
    # every enterprise exactly one product (all B-class). Pure forced reduction.
    a = np.diag([2.0, 3.0, 4.0])
    p = Problem(a=a, z=np.array([1.0, 1.0, 1.0]), b0=np.array([1.0, 1.0, 1.0]))
    oracle = FeasibilitySolver(p)
    assert oracle.is_feasible([2.0, 3.0, 4.0])      # exactly the capacities
    assert not oracle.is_feasible([2.0, 3.0, 4.001])  # just beyond


if __name__ == "__main__":
    from mib_tests.run_all import run_module

    run_module(__name__)
