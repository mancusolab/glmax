# pattern: Functional Core

from typing import NamedTuple

from ._fit import FittedGLM


__all__ = ["Diagnostics", "check"]


class Diagnostics(NamedTuple):
    """Model-fit diagnostics contract returned by `check(...)`.

    !!! note
        This is a placeholder contract. No diagnostic fields are computed yet.
        The seam is reserved for residuals, calibration, and influence diagnostics
        in a future release.
    """


def check(fitted: FittedGLM) -> Diagnostics:
    r"""Assess model fit and return a diagnostics noun.

    The canonical `check` grammar verb. Currently returns an empty `Diagnostics`
    placeholder; use as a seam for residual and calibration diagnostics.

    **Arguments:**

    - `fitted`: `FittedGLM` noun produced by `fit(...)`.

    **Returns:**

    `Diagnostics` noun (currently empty).

    **Raises:**

    - `TypeError`: if `fitted` is not a `FittedGLM` instance.
    """
    if not isinstance(fitted, FittedGLM):
        raise TypeError("check(...) expects `fitted` to be a FittedGLM instance.")

    return Diagnostics()
