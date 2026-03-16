# pattern: Functional Core

from __future__ import annotations

from typing import NamedTuple

from ..fit import _matches_fit_result_shape, _matches_fitted_glm_shape, FittedGLM, validate_fit_result


__all__ = ["Diagnostics", "check"]


class Diagnostics(NamedTuple):
    """Placeholder model-fit diagnostics contract returned by check()."""


def check(fitted: FittedGLM) -> Diagnostics:
    """Placeholder model-fit assessment seam over fit artifacts."""
    if not _matches_fitted_glm_shape(fitted):
        raise TypeError("check(...) expects `fitted` to be a FittedGLM instance.")
    if not _matches_fit_result_shape(fitted.result):
        raise TypeError("check(...) expects `fitted.result` to be a FitResult instance.")
    validate_fit_result(fitted.result)
    return Diagnostics()
