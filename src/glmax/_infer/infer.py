# pattern: Functional Core


import equinox as eqx

from .._fit import FittedGLM
from .hyptest import AbstractTest, WaldTest
from .stderr import AbstractStdErrEstimator, FisherInfoError
from .types import InferenceResult


__all__ = ["infer"]


@eqx.filter_jit
def infer(
    fitted: FittedGLM,
    inferrer: AbstractTest = WaldTest(),
    stderr: AbstractStdErrEstimator = FisherInfoError(),
) -> InferenceResult:
    r"""Compute inferential summaries from a fitted GLM without refitting.

    The canonical `infer` grammar verb. Delegates to the chosen
    [`glmax.AbstractTest`][] strategy, which calls the selected
    [`glmax.AbstractStdErrEstimator`][] as needed. Inference is computed from
    the fitted noun only; no model refit is performed inside
    [`glmax.infer`][].

    **Arguments:**

    - `fitted`: fitted [`glmax.FittedGLM`][] noun produced by
      [`glmax.fit`][].
    - `inferrer`: [`glmax.AbstractTest`][] strategy. Defaults to
      [`glmax.WaldTest`][].
    - `stderr`: [`glmax.AbstractStdErrEstimator`][] forwarded to the
      inferrer. Defaults to [`glmax.FisherInfoError`][].

    **Returns:**

    [`glmax.InferenceResult`][] carrying `(params, se, stat, p)`.

    **Raises:**

    - `TypeError`: if `fitted` is not a [`glmax.FittedGLM`][], `inferrer` is
      not a [`glmax.AbstractTest`][], or `stderr` is not a
      [`glmax.AbstractStdErrEstimator`][].
    """

    if not isinstance(fitted, FittedGLM):
        raise TypeError("infer(...) expects `fitted` to be a FittedGLM instance.")
    if not isinstance(inferrer, AbstractTest):
        raise TypeError("infer(...) expects `inferrer` to be an AbstractTest instance.")
    if not isinstance(stderr, AbstractStdErrEstimator):
        raise TypeError("infer(...) expects `stderr` to be an AbstractStdErrEstimator instance.")

    return inferrer(fitted, stderr)
