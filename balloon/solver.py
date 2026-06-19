"""Outer balloon-inflation loop: search for the optimal number of sets ``mu*``.

This implements the iterative procedure of the 1977 paper (points 1, 8 and
equations (9), (10), (26)-(31)).  The demands ``b0[j]*mu`` are inflated like a
balloon: ``mu`` is grown while a feasible allocation still exists and shrunk
(by dichotomy) once it does not, until the step falls below ``eps`` (eq. 9):

    mu_{t+1} = mu_t + alpha                                            (10)
    alpha = 0.1 * mu_t / 10**k         while feasible   (eq. 19 holds)  (30)
    alpha = -|mu_t - mu_{t-1}| / 2      on overshoot     (eq. 20/21)     (31)

``k`` counts the overshoots (dichotomy steps).  The starting pressure is the
paper's boundary estimate ``mu_gr`` (eq. 26-29) divided by ten:

    mu_gr = m / (n (n+m)) * sum_i max_j atilde[i][j],   atilde = a / b0    (29)
    mu_0  = mu_gr / 10                                                     (paper)

Because the feasibility oracle is *monotone* in ``mu`` (larger ``mu`` ->
larger demands -> harder), the search is bracketed by a known-feasible ``lo``
and a known-infeasible ``hi``; if the paper's schedule does not converge within
``max_iter`` steps we finish with plain bisection on that bracket, so the result
is always correct to ``eps``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .feasibility import FeasibilitySolver
from .problem import Allocation, Problem


@dataclass
class IterationRecord:
    """One step of the inflation trace (for inspection / the paper-style log)."""

    t: int
    mu: float
    feasible: bool
    step: float  # alpha used to reach the *next* mu (0 for the terminal step)
    kind: str    # "grow" (eq. 30) or "dichotomy" (eq. 31)


@dataclass
class BalloonResult:
    """Result of the uniform balloon inflation."""

    mu: float
    allocation: Allocation
    iterations: int
    history: list[IterationRecord] = field(default_factory=list)
    mu_gr: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.allocation is not None and self.allocation.satisfies()


class BalloonSolver:
    """Solve ``max mu`` s.t. demands ``b0*mu`` are feasible (Kantorovich ``mu*``).

    Parameters
    ----------
    problem:
        The problem to solve.
    eps:
        Convergence tolerance on ``mu`` (the ``epsilon`` of eq. 9).
    max_iter:
        Cap on inflation steps before falling back to bisection.
    max_nodes:
        Node budget forwarded to the default feasibility oracle.
    oracle:
        Feasibility oracle to query (any object exposing
        ``solve(demand) -> Allocation | None``).  Defaults to the faithful
        class-fixing :class:`~balloon.feasibility.FeasibilitySolver`; pass
        :class:`~balloon.exact.LPOracle` for a guaranteed-optimal search.
    """

    def __init__(self, problem: Problem, eps: float = 1e-4,
                 max_iter: int = 2000, max_nodes: int = 2_000_000,
                 oracle=None):
        self.p = problem
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.oracle = oracle if oracle is not None else FeasibilitySolver(
            problem, max_nodes=max_nodes
        )

    # ------------------------------------------------------------------ #
    def boundary_estimate(self) -> float:
        """The paper's boundary pressure ``mu_gr`` (eq. 26-29)."""
        m, n = self.p.m, self.p.n
        atilde = self.p.a / self.p.b0[None, :]  # a_ij = a'_ij / b_j0 (transform 6)
        return float(m / (n * (n + m)) * atilde.max(axis=1).sum())

    def _feasible(self, mu: float):
        """Return an allocation for demands ``b0*mu`` (or ``None``)."""
        if mu <= 0:
            # zero demand is always trivially feasible
            return self.oracle.solve(np.zeros(self.p.n))
        return self.oracle.solve(self.p.demand(mu))

    # ------------------------------------------------------------------ #
    def solve(self) -> BalloonResult:
        """Run the balloon inflation and return the optimal ``mu*``."""
        eps = self.eps
        history: list[IterationRecord] = []

        mu_gr = self.boundary_estimate()
        mu0 = mu_gr / 10.0
        if mu0 <= 0:
            mu0 = eps

        # --- bracket trackers: lo = best feasible, hi = lowest infeasible.
        lo, lo_alloc = 0.0, self._feasible(0.0)
        hi = float("inf")

        # The starting pressure must be feasible to begin "growing"; if the
        # heuristic mu0 already overshoots, treat it as the first infeasible hi.
        mu = mu0
        prev_mu = 0.0
        k = 0  # number of dichotomy (overshoot) steps -> exponent in eq. 30
        t = 0

        while t < self.max_iter:
            alloc = self._feasible(mu)
            feas = alloc is not None
            if feas:
                if mu > lo:
                    lo, lo_alloc = mu, alloc
                alpha = 0.1 * mu / (10 ** k)  # eq. (30)
                kind = "grow"
            else:
                hi = min(hi, mu)
                alpha = -abs(mu - prev_mu) / 2.0  # eq. (31) dichotomy
                k += 1
                kind = "dichotomy"

            history.append(IterationRecord(t=t, mu=mu, feasible=feas, step=alpha, kind=kind))
            t += 1

            # Convergence (eq. 9): the bracket (or the step) is below epsilon.
            if hi - lo <= eps or abs(alpha) <= eps:
                break

            nxt = mu + alpha
            # keep the iterate strictly inside the (lo, hi) bracket for safety
            if nxt <= lo:
                nxt = (lo + (hi if hi < float("inf") else lo + 2 * abs(alpha))) / 2.0
            if hi < float("inf") and nxt >= hi:
                nxt = (lo + hi) / 2.0
            prev_mu, mu = mu, nxt

        # --- robust finish: bisect the bracket if the schedule stalled.
        if hi == float("inf"):
            # never overshot: push upward geometrically to find an infeasible hi
            probe = max(lo, eps) if lo > 0 else max(mu0, eps)
            while probe <= 1e18:
                probe *= 2.0
                alloc = self._feasible(probe)
                feas = alloc is not None
                history.append(IterationRecord(t=t, mu=probe, feasible=feas,
                                               step=0.0, kind="probe"))
                t += 1
                if not feas:
                    hi = probe
                    break
                lo, lo_alloc = probe, alloc
        while hi - lo > eps:
            mid = (lo + hi) / 2.0
            alloc = self._feasible(mid)
            feas = alloc is not None
            if feas:
                lo, lo_alloc = mid, alloc
            else:
                hi = mid
            history.append(IterationRecord(t=t, mu=mid, feasible=feas,
                                           step=0.0, kind="bisect"))
            t += 1

        return BalloonResult(
            mu=lo,
            allocation=lo_alloc,
            iterations=t,
            history=history,
            mu_gr=mu_gr,
        )


def solve(problem: Problem, eps: float = 1e-4) -> BalloonResult:
    """Convenience wrapper around :class:`BalloonSolver`."""
    return BalloonSolver(problem, eps=eps).solve()
