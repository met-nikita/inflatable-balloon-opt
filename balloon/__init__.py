"""Inflatable Balloon Method (метод «надувного шарика», МНШ / MIB).

A pure-Python (numpy) implementation of G.B. Bronfeld's method for the optimal
distribution of a production plan -- Kantorovich's assortment / machine-loading
problem -- and its nonlinear multi-criteria generalisation.  See
the project ``README.md`` for the mapping between code and paper equations.

Quick start
-----------
>>> from balloon import Problem, BalloonSolver
>>> p = Problem.from_lists(a=[[2.0, 1.0], [1.0, 2.0]], z=[1.0, 1.0], b0=[1.0, 1.0])
>>> res = BalloonSolver(p).solve()
>>> round(res.mu, 2)
2.0
"""

from .problem import Allocation, Problem
from .feasibility import FeasibilitySolver, SearchBudgetExceeded, feasible
from .solver import BalloonResult, BalloonSolver, IterationRecord, solve
from .nonlinear import (
    GeneralizedBalloonSolver,
    GeneralizedResult,
    solve_generalized,
)


def __getattr__(name):  # lazy: keep scipy optional unless the exact API is used
    if name in ("LPOracle", "solve_exact", "feasible_exact", "ExactResult"):
        from . import exact

        return getattr(exact, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Problem",
    "Allocation",
    "FeasibilitySolver",
    "SearchBudgetExceeded",
    "feasible",
    "BalloonSolver",
    "BalloonResult",
    "IterationRecord",
    "solve",
    "GeneralizedBalloonSolver",
    "GeneralizedResult",
    "solve_generalized",
    # exact reference (scipy-backed, imported lazily)
    "LPOracle",
    "solve_exact",
    "feasible_exact",
]

__version__ = "0.1.0"
