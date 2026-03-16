# pattern: Functional Core

from __future__ import annotations

from .._fit import (
    FittedGLM,
)
from .hyptest import AbstractTest, WaldTest
from .stderr import AbstractStdErrEstimator, FisherInfoError
from .types import InferenceResult


__all__ = ["infer"]


def infer(
    fitted: FittedGLM,
    inferrer: AbstractTest = WaldTest(),
    stderr: AbstractStdErrEstimator = FisherInfoError(),
) -> InferenceResult:
    """Inferential summaries from fit artifacts without refitting.

    **Arguments:**

    - `fitted`: validated fitted-model carrier produced by `glmax.fit(...)`.
    - `inferrer`: optional inference strategy. `None` resolves lazily to
      `DEFAULT_INFERRER` at call time.
    - `stderr`: standard-error estimator forwarded to the selected inferrer.

    **Returns:**

    - `InferenceResult` carrying `(params, se, stat, p)`.

    **Raises:**

    - `TypeError`: if `fitted` is not a `FittedGLM`, `fitted.result` is not a
      `FitResult`, `inferrer` is not an `AbstractTest`, or `stderr` is not
      an `AbstractStdErrEstimator`.
    """

    if not isinstance(fitted, FittedGLM):
        raise TypeError("infer(...) expects `fitted` to be a FittedGLM instance.")
    if not isinstance(inferrer, AbstractTest):
        raise TypeError("infer(...) expects `inferrer` to be an AbstractTest instance.")
    if not isinstance(stderr, AbstractStdErrEstimator):
        raise TypeError("infer(...) expects `stderr` to be an AbstractStdErrEstimator instance.")

    return inferrer(fitted, stderr)
