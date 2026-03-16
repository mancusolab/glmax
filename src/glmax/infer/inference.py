# pattern: Functional Core

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp

from jax import Array
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from ..family.dist import ExponentialFamily, Gaussian
from ..family.utils import t_cdf
from ..fit import _matches_fit_result_shape, _matches_fitted_glm_shape, FittedGLM, Params, validate_fit_result
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
    stderr: AbstractStdErrEstimator = DEFAULT_STDERR,
) -> InferenceResult:
    """Inferential summaries from fit artifacts without refitting."""
    if not _matches_fitted_glm_shape(fitted):
        raise TypeError("infer(...) expects `fitted` to be a FittedGLM instance.")
    if not isinstance(stderr, AbstractStdErrEstimator):
        raise TypeError("infer(...) expects `stderr` to be an AbstractStdErrEstimator instance.")

    model = fitted.model
    fit_result = fitted.result
    if not _matches_fit_result_shape(fit_result):
        raise TypeError("infer(...) expects `fitted.result` to be a FitResult instance.")
    validate_fit_result(fit_result)

    beta = jnp.asarray(fit_result.params.beta)
    covariance = jnp.asarray(stderr(fitted))
    se = jnp.sqrt(jnp.diag(covariance))
    stat = beta / se
    df = int(fit_result.eta.shape[0] - beta.shape[0])
    p = wald_test(stat, df, model.family)

    return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)
