"""Exact reference solver via linear programming (``scipy.optimize.linprog``).

The Balloon Method is a from-scratch combinatorial algorithm; this module is an
*independent* ground truth used to validate it.  The optimal number of
assortment sets ``mu*`` is the value of the linear program

    maximize    mu
    subject to  sum_i a[i][j] x[i][j] >= b0[j] mu     (j = 1..n)
                sum_j x[i][j]         <= z[i]          (i = 1..m)
                x >= 0,  mu >= 0

which is exactly the problem the balloon inflates toward.  ``scipy`` is only a
dependency of this reference module and of the tests, never of the method
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog

from .problem import Allocation, Problem


@dataclass
class ExactResult:
    mu: float
    allocation: Allocation


def _var_index(problem: Problem):
    m, n = problem.m, problem.n
    # variables: x[i][j] flattened row-major, then mu as the last variable.
    n_x = m * n
    return n_x, n_x  # (count of x vars, index of mu)


def solve_exact(problem: Problem) -> ExactResult:
    """Compute the exact optimum ``mu*`` and a corresponding allocation."""
    m, n = problem.m, problem.n
    n_x, mu_idx = _var_index(problem)
    n_var = n_x + 1

    # objective: maximize mu == minimize -mu
    c = np.zeros(n_var)
    c[mu_idx] = -1.0

    rows, rhs = [], []
    # demand constraints (2): -sum_i a_ij x_ij + b0_j mu <= 0
    for j in range(n):
        row = np.zeros(n_var)
        for i in range(m):
            row[i * n + j] = -problem.a[i, j]
        row[mu_idx] = problem.b0[j]
        rows.append(row)
        rhs.append(0.0)
    # time constraints (3): sum_j x_ij <= z_i
    for i in range(m):
        row = np.zeros(n_var)
        for j in range(n):
            row[i * n + j] = 1.0
        rows.append(row)
        rhs.append(problem.z[i])

    A_ub = np.array(rows)
    b_ub = np.array(rhs)
    bounds = [(0.0, None)] * n_var

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")

    mu = float(res.x[mu_idx])
    x = res.x[:n_x].reshape(m, n)
    return ExactResult(mu=mu, allocation=Allocation(x=x, demand=problem.demand(mu), problem=problem))


class LPOracle:
    """Exact feasibility oracle (LP-backed), drop-in for :class:`FeasibilitySolver`.

    Pass it to :class:`~balloon.solver.BalloonSolver` (``oracle=LPOracle(p)``) to
    obtain the *guaranteed* optimum on any instance -- at the cost of replacing
    the paper's combinatorial inner method with a linear program.
    """

    def __init__(self, problem: Problem, **_ignored):
        self.p = problem

    def solve(self, demand, z: Optional[np.ndarray] = None) -> Optional[Allocation]:
        return feasible_exact(self.p, demand, z=z)

    def is_feasible(self, demand, z: Optional[np.ndarray] = None) -> bool:
        return feasible_exact(self.p, demand, z=z) is not None


def feasible_exact(problem: Problem, demand, z: Optional[np.ndarray] = None) -> Optional[Allocation]:
    """LP feasibility oracle for a fixed ``demand`` vector (ground truth)."""
    m, n = problem.m, problem.n
    demand = np.asarray(demand, dtype=float).ravel()
    zz = problem.z if z is None else np.asarray(z, dtype=float)
    n_var = m * n
    c = np.zeros(n_var)  # pure feasibility

    rows, rhs = [], []
    for j in range(n):  # -sum_i a_ij x_ij <= -demand_j
        row = np.zeros(n_var)
        for i in range(m):
            row[i * n + j] = -problem.a[i, j]
        rows.append(row)
        rhs.append(-demand[j])
    for i in range(m):  # sum_j x_ij <= z_i
        row = np.zeros(n_var)
        for j in range(n):
            row[i * n + j] = 1.0
        rows.append(row)
        rhs.append(zz[i])

    res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=[(0.0, None)] * n_var, method="highs")
    if not res.success:
        return None
    x = res.x.reshape(m, n)
    return Allocation(x=x, demand=demand, problem=problem)
