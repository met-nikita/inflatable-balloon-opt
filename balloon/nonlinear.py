"""Generalised "inflatable balloon" with nonlinear, per-criterion inflation.

The 2019 retrospective generalises the
classic problem.  Instead of every demand growing in lockstep as ``b0[j]*mu``,
each criterion gets its own desired level

    b[j] = f_j(b0[j], mu, t)                                              (8)

where ``f_j`` is a *monotone non-decreasing* "desired-level" function (Fig. 1-3).
Geometrically: the balloon first inflates uniformly until it touches the first
resource wall (the Kantorovich optimum ``mu*``); if resources remain, individual
criteria keep bulging into the free space until the whole resource is used
("ресурс полностью использован") or each criterion reaches its desired ceiling.

This module implements that two-phase scheme on top of the uniform solver:

1. **Uniform inflation** -> ``mu*`` and demands ``b0[j]*mu*`` (Kantorovich).
2. **Per-criterion inflation** -> round-robin, raise each ``b[j]`` as far as the
   shared resources (and an optional ceiling ``cap[j]``) allow, re-checking full
   feasibility every time so already-granted criteria are never violated.

Note that the phase-2 outcome is a point on the Pareto frontier and therefore
depends on the inflation order / ceilings - exactly the knobs the nonlinear
target functions are meant to express.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .feasibility import FeasibilitySolver
from .problem import Allocation, Problem
from .solver import BalloonSolver

# A desired-level function f_j(b0_j, mu, t) -> desired demand for product j.
TargetFn = Callable[[float, float, int], float]


@dataclass
class GeneralizedResult:
    mu: float                 # uniform Kantorovich optimum (end of phase 1)
    demand: np.ndarray        # final inflated demand levels b[j] (the b_jmax)
    allocation: Allocation    # feasible allocation realising `demand`
    rounds: int               # phase-2 round-robin passes performed
    base_demand: np.ndarray = field(default_factory=lambda: np.array([]))
    # phase-2 snapshots: one per successful per-criterion inflation, each
    # {"moved": j, "demand": [...]}, the first being the uniform mu* demand.
    history: list = field(default_factory=list)

    @property
    def extra_sets(self) -> np.ndarray:
        """How many *extra* per-product sets phase 2 won over the uniform mu*."""
        return self.demand / self.allocation.problem.b0 - self.mu


class GeneralizedBalloonSolver:
    """Two-phase generalised balloon solver.

    Parameters
    ----------
    problem:
        The problem to solve.
    eps:
        Tolerance for both phases.
    caps:
        Optional ``(n,)`` absolute ceilings on the demand of each product
        (the maximum desired level).  ``None``/``inf`` means "inflate until the
        resources stop you".
    order:
        Optional product-index order for phase-2 inflation (priority).  Defaults
        to ascending remaining-capacity (tightest products first).
    max_rounds:
        Safety cap on phase-2 round-robin passes.
    """

    def __init__(self, problem: Problem, eps: float = 1e-4,
                 caps: Optional[Sequence[float]] = None,
                 order: Optional[Sequence[int]] = None,
                 max_rounds: int = 50, max_nodes: int = 2_000_000,
                 oracle=None):
        self.p = problem
        self.eps = float(eps)
        self.caps = (np.full(problem.n, np.inf) if caps is None
                     else np.asarray(caps, dtype=float))
        self.order = list(order) if order is not None else None
        self.max_rounds = int(max_rounds)
        self.oracle = oracle if oracle is not None else FeasibilitySolver(
            problem, max_nodes=max_nodes
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_target_functions(cls, problem: Problem, targets: Sequence[TargetFn],
                              mu: float, t: int = 0, **kwargs) -> "GeneralizedBalloonSolver":
        """Build a solver whose ceilings come from desired-level functions ``f_j`` (eq. 8)."""
        caps = np.array([targets[j](float(problem.b0[j]), float(mu), t)
                         for j in range(problem.n)], dtype=float)
        return cls(problem, caps=caps, **kwargs)

    # ------------------------------------------------------------------ #
    def solve(self) -> GeneralizedResult:
        # ---- phase 1: uniform inflation to the Kantorovich optimum mu*.
        base = BalloonSolver(self.p, eps=self.eps, oracle=self.oracle).solve()
        demand = self.p.demand(base.mu).copy()
        base_demand = demand.copy()

        # respect ceilings even at the uniform level
        demand = np.minimum(demand, self.caps)

        # ---- phase 2: per-criterion inflation into the leftover resources.
        order = self.order if self.order is not None else self._default_order(demand)
        history = [{"moved": -1, "demand": demand.tolist()}]  # uniform mu* shape
        rounds = 0
        improved = True
        while improved and rounds < self.max_rounds:
            improved = False
            for j in order:
                gained = self._inflate_one(demand, j)
                if gained > self.eps:
                    improved = True
                    history.append({"moved": int(j), "demand": demand.tolist()})
            rounds += 1

        alloc = self.oracle.solve(demand)
        if alloc is None:  # should not happen; demand stays feasible by construction
            alloc = self.oracle.solve(base_demand)
        return GeneralizedResult(
            mu=base.mu,
            demand=demand,
            allocation=alloc,
            rounds=rounds,
            base_demand=base_demand,
            history=history,
        )

    # ------------------------------------------------------------------ #
    def _default_order(self, demand: np.ndarray) -> list[int]:
        """Inflate the products with the least spare column capacity first."""
        spare = (self.p.a * self.p.z[:, None]).sum(axis=0) - demand
        return list(np.argsort(spare))

    def _inflate_one(self, demand: np.ndarray, j: int) -> float:
        """Raise ``demand[j]`` to the largest feasible value <= ``cap[j]``.

        Returns the gained amount.  All other demands are held fixed and full
        feasibility is re-verified, so previously satisfied criteria are kept.
        """
        cap = self.caps[j]
        start = demand[j]
        if cap - start <= self.eps:
            return 0.0

        trial = demand.copy()

        def ok(value: float) -> bool:
            trial[j] = value
            return self.oracle.is_feasible(trial)

        # geometric search for an infeasible upper bound (bounded by cap).
        lo = start
        # first probe step relative to current level (or column capacity)
        col_cap = float((self.p.a[:, j] * self.p.z).sum())
        step = max(self.eps, 0.1 * max(start, col_cap))
        hi = min(start + step, cap)
        while ok(hi) and hi < cap - self.eps:
            lo = hi
            step *= 2.0
            hi = min(hi + step, cap)
        if ok(hi):  # reached the ceiling and it is feasible
            demand[j] = hi
            return hi - start

        # bisect between feasible lo and infeasible hi
        while hi - lo > self.eps:
            mid = (lo + hi) / 2.0
            if ok(mid):
                lo = mid
            else:
                hi = mid
        demand[j] = lo
        return lo - start


def solve_generalized(problem: Problem, eps: float = 1e-4,
                      caps: Optional[Sequence[float]] = None) -> GeneralizedResult:
    """Convenience wrapper around :class:`GeneralizedBalloonSolver`."""
    return GeneralizedBalloonSolver(problem, eps=eps, caps=caps).solve()
