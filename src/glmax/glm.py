# pattern: Functional Core

"""GLM specification noun and grammar verb factory."""

from __future__ import annotations

import equinox as eqx

from .family.dist import ExponentialFamily, Gaussian


class GLM(eqx.Module):
    r"""Generalized Linear Model specification noun.

    A `GLM` holds the family that characterises a model. It carries no state
    (no fitted parameters, no data) and is constructed once via `specify`.
    The linear solver is part of the `AbstractFitter` strategy, not the model.

    **Arguments:**

    - `family`: `ExponentialFamily` instance (default: `Gaussian()`).
    """

    family: ExponentialFamily = Gaussian()


def specify(
    family: ExponentialFamily = Gaussian(),
) -> GLM:
    r"""Construct a GLM specification.

    **Arguments:**

    - `family`: `ExponentialFamily` distribution (default: `Gaussian()`).

    **Returns:**

    A `GLM` specification noun.
    """
    return GLM(family=family)
