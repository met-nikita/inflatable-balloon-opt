"""Inner feasibility oracle of the Inflatable Balloon Method.

Given a :class:`~balloon.problem.Problem` and a fixed demand vector
``b[j]`` (the right-hand sides of constraint (6)/(2)), decide whether a feasible
time allocation exists

    exists x >= 0 with   sum_i a[i][j] x[i][j] >= b[j]   for all j     (6)
                         sum_j x[i][j]         <= z[i]   for all i     (7)

and return one such allocation if it does.

This is the "search for a feasible solution ``[x_ij]*`` on one iteration"
described in the 1977 paper.  It is the *oracle* the outer balloon-inflation
loop (see :mod:`balloon.solver`) queries while growing ``mu``.

The method (Sections "ОПИСАНИЕ АЛГОРИТМА", points 1-8)
-----------------------------------------------------
Estimates (eq. 11)         ``s[i][j] = a[i][j] * d[i][j]`` where ``d[i][j]`` is
                           the working time still available to variable
                           ``x[i][j]``.  Initially ``d[i][j] = z[i]`` for every
                           ``a[i][j] > 0`` (the paper normalises ``z_i = 1`` and
                           writes ``d = 1``; we keep the general resource), so
                           ``s[i][j]`` is the *maximum output* of product *j*
                           enterprise *i* could deliver with all of its time.

Estimate classes:
  * **A-class** (def. 1)   product column with a *single* positive estimate:
                           that enterprise is the only possible supplier -> its
                           assignment is *forced* (eq. 15).
  * **B-class** (def. 2)   enterprise row with a *single* positive estimate:
                           the enterprise can make only one (still-needed)
                           product -> assign it there (eq. 16).  Always safe.
  * **C-class** (def. 3)   the largest estimate in its column (a strict
                           column-maximum) -- the "maximum element" heuristic.
  * **D-class** (def. 4)   otherwise the global maximum estimate (eq. 14).

A/B are deterministic, never-wrong reductions.  When only C/D choices remain we
*branch*: pick the most constrained still-unsatisfied product and try its
candidate suppliers in max-element order, recursing (branch & bound, point 4).
Branches are pruned with the necessary bound ``P`` (eq. 17) and the per-product
bound ``Q`` (eq. 18).

Fixing an assignment ``(k, l)`` -- "закрепление оценки" -- means committing some
of enterprise *k*'s time to product *l* (eq. 15/22-25):

  * if ``s[k][l] >= b[l]``  enterprise *k* alone covers the remaining demand of
    *l*; it spends exactly ``b[l]/a[k][l]`` time, product *l* is done, and *k*
    keeps its leftover time for other products;
  * otherwise *k* pours *all* of its remaining time into *l* (still short), and
    the rest of the demand must come from other enterprises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .problem import TOL, Allocation, Problem


class SearchBudgetExceeded(RuntimeError):
    """Raised when the branch-and-bound search exceeds its node budget.

    The feasibility test is complete but worst-case exponential; this guards
    against pathological instances rather than indicating infeasibility.
    """


@dataclass
class _State:
    """Mutable working state of one branch of the feasibility search."""

    avail: np.ndarray  # (m,)  remaining time per enterprise   (d, eq. 11)
    rem: np.ndarray    # (n,)  remaining demand per product     (b_vt_j, point 2)
    x: np.ndarray      # (m,n) committed times                  (the d^{WR}_ij, eq. 37)

    def copy(self) -> "_State":
        return _State(self.avail.copy(), self.rem.copy(), self.x.copy())


class FeasibilitySolver:
    """Branch-and-bound feasibility oracle for a single demand vector.

    Parameters
    ----------
    problem:
        The problem providing the productivity matrix ``a`` and (default)
        resources ``z``.
    tol:
        Numerical tolerance.
    max_nodes:
        Safety cap on branch-and-bound nodes; exceeding it raises
        :class:`SearchBudgetExceeded`.
    """

    def __init__(self, problem: Problem, tol: float = 1e-9, max_nodes: int = 2_000_000):
        self.p = problem
        self.a = problem.a
        self.tol = tol
        self.max_nodes = max_nodes
        self._nodes = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def solve(self, demand, z: Optional[np.ndarray] = None) -> Optional[Allocation]:
        """Return a feasible :class:`Allocation` for ``demand`` or ``None``.

        ``z`` overrides the enterprise resources (used by the generalised
        balloon when some resource has already been spent); defaults to the
        problem's ``z``.
        """
        demand = np.asarray(demand, dtype=float).ravel()
        if demand.shape != (self.p.n,):
            raise ValueError("demand must have length n")
        avail = (self.p.z if z is None else np.asarray(z, dtype=float)).copy()
        self._nodes = 0
        state = _State(
            avail=avail,
            rem=np.maximum(demand, 0.0),
            x=np.zeros_like(self.a),
        )
        result = self._search(state)
        if result is None:
            return None
        return Allocation(x=result.x, demand=demand, problem=self.p)

    def is_feasible(self, demand, z: Optional[np.ndarray] = None) -> bool:
        """Boolean convenience wrapper around :meth:`solve`."""
        return self.solve(demand, z=z) is not None

    # ------------------------------------------------------------------ #
    # Core recursion
    # ------------------------------------------------------------------ #
    def _search(self, st: _State) -> Optional[_State]:
        self._nodes += 1
        if self._nodes > self.max_nodes:
            raise SearchBudgetExceeded(
                f"branch-and-bound exceeded {self.max_nodes} nodes"
            )

        # (points 2,3,5) apply all forced A/B reductions; may prove infeasible.
        if not self._apply_forced(st):
            return None

        active = self._active_products(st)
        if active.size == 0:
            return st  # all demands met -> feasible allocation found

        # (point 4) prune with the necessary bounds P (eq. 17) and Q (eq. 18).
        if not self._bounds_ok(st, active):
            return None

        # (points 6,7) no forced move left: branch on the hardest product,
        # trying its suppliers in max-element (C/D-class) order.
        l = self._choose_product(st, active)
        for k in self._candidate_suppliers(st, l):
            child = st.copy()
            self._fix(child, k, l)
            found = self._search(child)
            if found is not None:
                return found
        return None

    # ------------------------------------------------------------------ #
    # Forced reductions: A-class (eq. 15) and B-class (eq. 16)
    # ------------------------------------------------------------------ #
    def _apply_forced(self, st: _State) -> bool:
        """Repeatedly apply A/B-class fixings.  Returns False if infeasible."""
        a = self.a
        tol = self.tol
        changed = True
        while changed:
            changed = False

            active = self._active_products(st)
            if active.size == 0:
                return True

            # --- A-class (def. 1): a product with exactly one possible supplier.
            for l in active:
                suppliers = np.where((a[:, l] > tol) & (st.avail > tol))[0]
                if suppliers.size == 0:
                    return False  # product needed but nobody can make it -> infeasible
                if suppliers.size == 1:
                    self._fix(st, int(suppliers[0]), int(l))
                    changed = True
                    break
            if changed:
                continue

            # --- B-class (def. 2): an enterprise that can serve only one product.
            for k in np.where(st.avail > tol)[0]:
                opts = np.where((a[k, :] > tol) & (st.rem > tol))[0]
                if opts.size == 1:
                    self._fix(st, int(k), int(opts[0]))
                    changed = True
                    break
                if opts.size == 0:
                    # enterprise is useless for the remaining demand; retire it
                    # so it stops being re-examined (does not affect feasibility).
                    st.avail[k] = 0.0
        return True

    # ------------------------------------------------------------------ #
    # Branch-and-bound bounds: P (eq. 17, necessary) and Q (eq. 18, per column)
    # ------------------------------------------------------------------ #
    def _bounds_ok(self, st: _State, active: np.ndarray) -> bool:
        a = self.a
        tol = self.tol

        # Q_j (eq. 18): even if *every* enterprise poured all its time into j,
        # could demand j be met?  sum_i a[i][j]*avail[i] >= rem[j].
        col_capacity = (a[:, active] * st.avail[:, None]).sum(axis=0)
        if np.any(col_capacity < st.rem[active] - tol):
            return False

        # P (eq. 17): each enterprise can contribute at most its single best
        # product's worth, so total weighted output is bounded by
        # sum_i max_j a[i][j]*avail[i].  If that is below total remaining demand
        # the instance is infeasible (necessary, not sufficient).
        per_enterprise_best = (a[:, active] * st.avail[:, None]).max(axis=1)
        p = float(per_enterprise_best.sum() - st.rem[active].sum())
        if p < -tol:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Branching choices
    # ------------------------------------------------------------------ #
    def _choose_product(self, st: _State, active: np.ndarray) -> int:
        """Pick the most constrained unsatisfied product (smallest Q slack)."""
        a = self.a
        col_capacity = (a[:, active] * st.avail[:, None]).sum(axis=0)
        slack = col_capacity - st.rem[active]
        return int(active[int(np.argmin(slack))])

    def _candidate_suppliers(self, st: _State, l: int) -> list[int]:
        """Suppliers of product ``l`` ordered by estimate ``s[i][l]`` desc.

        This realises the C/D-class "maximum element" preference (def. 3/4):
        try the enterprise that can produce the most of ``l`` first.
        """
        a = self.a
        tol = self.tol
        ids = np.where((a[:, l] > tol) & (st.avail > tol))[0]
        est = a[ids, l] * st.avail[ids]  # s[i][l] (eq. 11)
        order = np.argsort(-est)
        return [int(ids[o]) for o in order]

    # ------------------------------------------------------------------ #
    # Fixing an assignment (eq. 15 / 22-25)
    # ------------------------------------------------------------------ #
    def _fix(self, st: _State, k: int, l: int) -> None:
        """Commit enterprise ``k``'s time to product ``l`` ("закрепление")."""
        a_kl = self.a[k, l]
        potential = a_kl * st.avail[k]  # s[k][l] = a[k][l]*d[k][l] (eq. 11)
        rem_l = st.rem[l]

        if potential >= rem_l - self.tol:
            # k alone covers the remaining demand of l (eq. 15, second branch).
            time_used = rem_l / a_kl
            st.x[k, l] += time_used
            st.avail[k] = max(st.avail[k] - time_used, 0.0)
            st.rem[l] = 0.0  # product l satisfied; column drops out (eq. 23/25)
        else:
            # k pours all remaining time into l, still short (eq. 15, first branch).
            st.x[k, l] += st.avail[k]
            st.rem[l] = rem_l - potential
            st.avail[k] = 0.0  # enterprise exhausted; row drops out (eq. 24/25)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _active_products(self, st: _State) -> np.ndarray:
        """Indices of products whose demand is not yet met."""
        return np.where(st.rem > self.tol)[0]


def feasible(problem: Problem, demand, z: Optional[np.ndarray] = None,
             max_nodes: int = 2_000_000) -> Optional[Allocation]:
    """Convenience wrapper: feasible allocation for ``demand`` or ``None``."""
    return FeasibilitySolver(problem, max_nodes=max_nodes).solve(demand, z=z)
