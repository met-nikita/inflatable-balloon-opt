"""Shared test helpers: random problem generator and LP-direction scaling."""

from __future__ import annotations

import numpy as np

from balloon import Problem
from balloon.exact import solve_exact


def random_problem(rng: np.random.Generator, m_range=(3, 6), n_range=(3, 6),
                   density=0.3) -> Problem:
    """A random feasible-by-construction problem.

    Every product has at least one supplier and every enterprise can make at
    least one product, so the optimum is well defined and non-trivial.  The
    default ``density`` (0.3) keeps the productivity matrix *sparse* -- the
    regime in which the class-fixing method is exact (see the README).
    """
    m = int(rng.integers(*m_range))
    n = int(rng.integers(*n_range))
    a = np.where(rng.random((m, n)) < density, rng.uniform(0.5, 2.0, (m, n)), 0.0)

    # guarantee at least one supplier per product (column) ...
    for j in range(n):
        if not np.any(a[:, j] > 0):
            a[int(rng.integers(m)), j] = rng.uniform(0.5, 2.0)
    # ... and at least one product per enterprise (row).
    for i in range(m):
        if not np.any(a[i, :] > 0):
            a[i, int(rng.integers(n))] = rng.uniform(0.5, 2.0)

    z = rng.uniform(1.0, 5.0, m)
    b0 = rng.uniform(1.0, 5.0, n)
    return Problem(a=a, z=z, b0=b0)


def max_scale(problem: Problem, direction: np.ndarray) -> float:
    """Largest ``t`` such that demand ``t*direction`` is LP-feasible (ground truth)."""
    direction = np.asarray(direction, dtype=float)
    probe = Problem(a=problem.a, z=problem.z, b0=np.maximum(direction, 1e-9))
    return solve_exact(probe).mu
