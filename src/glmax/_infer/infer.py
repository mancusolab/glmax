# pattern: Functional Core

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx

from .._fit import FittedGLM
from .hyptest import AbstractTest, WaldTest
from .stderr import AbstractStdErrEstimator, FisherInfoError


if TYPE_CHECKING:
    from .types import InferenceResult


__all__ = ["infer"]


@eqx.filter_jit
def infer(
    fitted: FittedGLM,
    inferrer: AbstractTest = WaldTest(),
    stderr: AbstractStdErrEstimator = FisherInfoError(),
) -> InferenceResult:
    r"""Compute inferential summaries from a fitted GLM without refitting.

    The canonical `infer` grammar verb. Delegates to the chosen `AbstractTest`
    strategy, which calls the `AbstractStdErrEstimator` as needed.

    **Arguments:**

    - `fitted`: `FittedGLM` noun produced by `fit(...)`.
    - `inferrer`: inference strategy (default: `WaldTest()`).
    - `stderr`: standard-error estimator forwarded to the inferrer
      (default: `FisherInfoError()`).

    **Returns:**

    `InferenceResult` carrying `(params, se, stat, p)`.

    **Raises:**

    - `TypeError`: if `fitted` is not a `FittedGLM`, `inferrer` is not an
      `AbstractTest`, or `stderr` is not an `AbstractStdErrEstimator`.
    """

    if not isinstance(fitted, FittedGLM):
        raise TypeError("infer(...) expects `fitted` to be a FittedGLM instance.")
    if not isinstance(inferrer, AbstractTest):
        raise TypeError("infer(...) expects `inferrer` to be an AbstractTest instance.")
    if not isinstance(stderr, AbstractStdErrEstimator):
        raise TypeError("infer(...) expects `stderr` to be an AbstractStdErrEstimator instance.")

    return inferrer(fitted, stderr)
