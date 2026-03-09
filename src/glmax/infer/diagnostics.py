# pattern: Functional Core

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

from jax import Array

from ..fit import FitResult, validate_fit_result


if TYPE_CHECKING:
    from ..glm import GLM


__all__ = ["Diagnostics", "check"]


class Diagnostics(NamedTuple):
    """Common diagnostics emitted by fit/check verbs."""

    converged: Array
    num_iters: Array
    objective: Array
    objective_delta: Array


def check(model: GLM, fit_result: FitResult) -> Diagnostics:
    """Diagnostics summaries from fit artifacts without refitting."""
    from ..glm import GLM as _GLM

    if not isinstance(model, _GLM):
        raise TypeError("check(...) expects `model` to be a GLM instance.")
    if not isinstance(fit_result, FitResult):
        raise TypeError("check(...) expects `fit_result` to be a FitResult instance.")

    validate_fit_result(fit_result)
    return fit_result.diagnostics
