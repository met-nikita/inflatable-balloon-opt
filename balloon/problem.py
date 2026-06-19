"""Problem model for the Inflatable Balloon Method (МНШ / MIB).

This module defines the optimisation problem solved by G.B. Bronfeld's
"Inflatable Balloon Method" (метод «надувного шарика»).

Setting (Kantorovich's assortment / machine-loading problem)
------------------------------------------------------------
* ``I = {1..m}``  -- enterprises (aggregates / machines / workshops).
* ``J = {1..n}``  -- product types that must be produced in a fixed assortment.
* ``a[i][j]``     -- productivity of enterprise *i* on product *j*
                     (output per unit of working time); ``a[i][j] == 0`` means
                     enterprise *i* cannot make product *j*.  This is ``a'_ij``
                     in the paper.
* ``z[i]``        -- total working-time resource of enterprise *i*
                     (the right-hand side ``z_i`` of equation (3)).
* ``b0[j]``       -- the assortment requirement of product *j*: one complete
                     assortment set needs ``b0[j]`` units of product *j*
                     (the vector ``{b_j0}``).

Decision variables ``x[i][j]`` are the *times* enterprise *i* spends on product
*j*.  We look for the largest number ``mu`` of complete assortment sets:

    maximize    mu
    subject to  sum_i a[i][j] * x[i][j]  >=  b0[j] * mu     for every product j   (2)
                sum_j x[i][j]            <=  z[i]           for every enterprise i (3)
                x[i][j] >= 0                                                       (4)

The Balloon Method inflates ``mu`` (and, in the generalised variant, the
individual demand levels) like air pressure until the "balloon" of demands
``b0[j]*mu`` presses against the resource walls.

Equation numbers in the docstrings refer to the 1977 paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

TOL = 1e-9  # numerical tolerance used throughout the package


@dataclass(frozen=True)
class Problem:
    """An instance of the optimal production-plan distribution problem.

    Attributes
    ----------
    a:
        ``(m, n)`` productivity matrix ``a[i][j] >= 0``.
    z:
        ``(m,)`` working-time resource per enterprise, ``z[i] > 0``.
    b0:
        ``(n,)`` assortment requirement per product, ``b0[j] > 0``.
    enterprises, products:
        Optional human-readable labels (default ``E0..`` / ``P0..``).
    """

    a: np.ndarray
    z: np.ndarray
    b0: np.ndarray
    enterprises: tuple[str, ...] = field(default=())
    products: tuple[str, ...] = field(default=())

    # ------------------------------------------------------------------ #
    # Construction / validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        a = np.asarray(self.a, dtype=float)
        z = np.asarray(self.z, dtype=float).ravel()
        b0 = np.asarray(self.b0, dtype=float).ravel()

        if a.ndim != 2:
            raise ValueError("a must be a 2-D matrix (m x n)")
        m, n = a.shape
        if z.shape != (m,):
            raise ValueError(f"z must have length m={m}, got {z.shape}")
        if b0.shape != (n,):
            raise ValueError(f"b0 must have length n={n}, got {b0.shape}")
        if np.any(a < -TOL):
            raise ValueError("productivities a[i][j] must be non-negative (5)")
        if np.any(z < -TOL):
            raise ValueError("resources z[i] must be non-negative")
        if np.any(b0 <= TOL):
            raise ValueError("assortment requirements b0[j] must be > 0 (5)")

        enterprises = tuple(self.enterprises) or tuple(f"E{i}" for i in range(m))
        products = tuple(self.products) or tuple(f"P{j}" for j in range(n))
        if len(enterprises) != m:
            raise ValueError("enterprises label count must equal m")
        if len(products) != n:
            raise ValueError("products label count must equal n")

        # frozen dataclass: bypass the setattr guard once, during init.
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "b0", b0)
        object.__setattr__(self, "enterprises", enterprises)
        object.__setattr__(self, "products", products)

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def m(self) -> int:
        """Number of enterprises."""
        return self.a.shape[0]

    @property
    def n(self) -> int:
        """Number of products."""
        return self.a.shape[1]

    def demand(self, mu: float) -> np.ndarray:
        """Demand levels ``b0[j] * mu`` for a given number of sets ``mu`` (eq. 2)."""
        return self.b0 * float(mu)

    # ------------------------------------------------------------------ #
    # Feasibility / evaluation of an allocation
    # ------------------------------------------------------------------ #
    def output(self, x: np.ndarray) -> np.ndarray:
        """Output per product produced by allocation ``x``: ``sum_i a[i][j] x[i][j]``."""
        return np.einsum("ij,ij->j", self.a, np.asarray(x, dtype=float))

    def sets_completed(self, x: np.ndarray) -> float:
        """Number of *complete* assortment sets an allocation delivers.

        A complete set needs ``b0[j]`` of every product, so the number of sets
        is limited by the scarcest product: ``min_j output_j / b0_j``.
        """
        out = self.output(x)
        return float(np.min(out / self.b0))

    def is_feasible(self, x: np.ndarray, demand: Sequence[float], tol: float = 1e-6) -> bool:
        """Check the constraints (2)-(4) for allocation ``x`` against ``demand``."""
        x = np.asarray(x, dtype=float)
        demand = np.asarray(demand, dtype=float)
        if np.any(x < -tol):
            return False
        if np.any(x.sum(axis=1) > self.z + tol):  # (3) time budget
            return False
        if np.any(self.output(x) < demand - tol):  # (2) demand satisfied
            return False
        return True

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
    @classmethod
    def from_lists(
        cls,
        a: Sequence[Sequence[float]],
        z: Sequence[float],
        b0: Sequence[float],
        enterprises: Sequence[str] = (),
        products: Sequence[str] = (),
    ) -> "Problem":
        """Build a :class:`Problem` from plain Python lists."""
        return cls(
            a=np.array(a, dtype=float),
            z=np.array(z, dtype=float),
            b0=np.array(b0, dtype=float),
            enterprises=tuple(enterprises),
            products=tuple(products),
        )


@dataclass
class Allocation:
    """A feasible (or candidate) time allocation produced by the solver.

    Attributes
    ----------
    x:
        ``(m, n)`` matrix of working times ``x[i][j]``.
    demand:
        The demand vector this allocation was built to satisfy.
    problem:
        Back-reference to the originating :class:`Problem`.
    """

    x: np.ndarray
    demand: np.ndarray
    problem: Problem

    @property
    def output(self) -> np.ndarray:
        """Output per product (``sum_i a[i][j] x[i][j]``)."""
        return self.problem.output(self.x)

    @property
    def used_time(self) -> np.ndarray:
        """Time used per enterprise (``sum_j x[i][j]``)."""
        return self.x.sum(axis=1)

    @property
    def slack_time(self) -> np.ndarray:
        """Unused time per enterprise (``z[i] - sum_j x[i][j]``)."""
        return self.problem.z - self.used_time

    @property
    def sets_completed(self) -> float:
        """Number of complete assortment sets delivered."""
        return self.problem.sets_completed(self.x)

    def satisfies(self, tol: float = 1e-6) -> bool:
        """True iff the allocation satisfies the problem constraints for ``demand``."""
        return self.problem.is_feasible(self.x, self.demand, tol=tol)

    def nonzero(self, tol: float = 1e-9) -> dict[tuple[str, str], float]:
        """Mapping of ``(enterprise, product) -> time`` for the non-zero entries."""
        out: dict[tuple[str, str], float] = {}
        for i in range(self.problem.m):
            for j in range(self.problem.n):
                if self.x[i, j] > tol:
                    out[(self.problem.enterprises[i], self.problem.products[j])] = float(
                        self.x[i, j]
                    )
        return out
