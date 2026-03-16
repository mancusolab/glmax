# pattern: Functional Core

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp

from jax import Array
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from ..family.dist import ExponentialFamily, Gaussian
from ..family.utils import t_cdf
from ..fit import _matches_fit_result_shape, _matches_fitted_glm_shape, FittedGLM, Params
from .stderr import AbstractStdErrEstimator, FisherInfoError


if TYPE_CHECKING:
    pass


__all__ = ["InferenceResult", "infer", "wald_test"]


DEFAULT_STDERR: AbstractStdErrEstimator = FisherInfoError()


class InferenceResult(NamedTuple):
    """Canonical infer verb output contract."""

    params: Params
    se: Array
    stat: Array
    p: Array


def wald_test(statistic: ArrayLike, df: int, family: ExponentialFamily) -> Array:
    r"""Two-sided Wald test p-values.

    Uses a $t_{df}$ distribution for Gaussian families and $\mathcal{N}(0, 1)$
    for all others.

    **Arguments:**

    - `statistic`: test statistics $\hat\beta / \mathrm{SE}(\hat\beta)$, shape `(p,)`.
    - `df`: residual degrees of freedom $n - p$.
    - `family`: fitted `ExponentialFamily` instance.

    **Returns:**

    Two-sided p-values, shape `(p,)`.
    """
    if isinstance(family, Gaussian):
        return 2 * t_cdf(-jnp.abs(statistic), df)
    return 2 * norm.sf(jnp.abs(statistic))


def infer(
    fitted: FittedGLM,
    inferrer=None,
    stderr: AbstractStdErrEstimator = DEFAULT_STDERR,
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
      `FitResult`, `inferrer` is not an `AbstractInferrer`, or `stderr` is not
      an `AbstractStdErrEstimator`.
    """
    from .inferrer import AbstractInferrer as _AbstractInferrer, DEFAULT_INFERRER as _DEFAULT_INFERRER

    if inferrer is None:
        inferrer = _DEFAULT_INFERRER

    if not _matches_fitted_glm_shape(fitted):
        raise TypeError("infer(...) expects `fitted` to be a FittedGLM instance.")
    if not _matches_fit_result_shape(fitted.result):
        raise TypeError("infer(...) expects `fitted.result` to be a FitResult instance.")
    if not isinstance(inferrer, _AbstractInferrer):
        raise TypeError("infer(...) expects `inferrer` to be an AbstractInferrer instance.")
    if not isinstance(stderr, AbstractStdErrEstimator):
        raise TypeError("infer(...) expects `stderr` to be an AbstractStdErrEstimator instance.")

    return inferrer(fitted, stderr)
