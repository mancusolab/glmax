# pattern: Functional Core

"""GLM specification noun and grammar verb factory."""

from __future__ import annotations

import equinox as eqx

from glmax._fit.solve import AbstractLinearSolver, CholeskySolver

from .family.dist import ExponentialFamily, Gaussian


class GLM(eqx.Module):
    r"""Generalized Linear Model specification noun.

    A `GLM` holds the family and solver that characterise a model. It carries
    no state (no fitted parameters, no data) and is constructed once via
    `specify`.

    **Arguments:**

    - `family`: `ExponentialFamily` instance (default: `Gaussian()`).
    - `solver`: `AbstractLinearSolver` instance (default: `CholeskySolver()`).
    """

    family: ExponentialFamily = Gaussian()
    solver: AbstractLinearSolver = CholeskySolver()


def specify(
    family: ExponentialFamily = Gaussian(),
    solver: AbstractLinearSolver = CholeskySolver(),
) -> GLM:
    r"""Construct a GLM specification.

    **Arguments:**

    - `family`: `ExponentialFamily` distribution (default: `Gaussian()`).
    - `solver`: `AbstractLinearSolver` for the IRLS normal equations (default: `CholeskySolver()`).

    **Returns:**

    A `GLM` specification noun.
    """
    return GLM(family=family, solver=solver)
