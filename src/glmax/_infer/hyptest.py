# pattern: Functional Core

"""Inferrer strategies for the `_infer()` verb."""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from .._fit import FittedGLM
from ..family.dist import Gaussian
from ..family.utils import t_cdf
from ..glm import GLM
from .stderr import _validated_fitted_dispersion, AbstractStdErrEstimator
from .types import InferenceResult


__all__ = ["AbstractTest", "WaldTest", "ScoreTest"]


class AbstractTest(eqx.Module, strict=True):
    r"""Abstract base for inference strategies used by `infer(fitted, inferrer=...)`.

    Subclasses implement `__call__` to compute test statistics and p-values
    from a [`glmax.FittedGLM`][]. The `stderr` estimator is passed in so
    strategies can choose whether to use it.
    """

    @abstractmethod
    def __call__(
        self,
        fitted: FittedGLM,
        stderr: AbstractStdErrEstimator,
    ) -> InferenceResult:
        r"""Compute inferential summaries from a fitted GLM.

        Concrete strategies may use the injected covariance estimator or
        ignore it if the statistic is computed directly from fit artifacts.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.
        - `stderr`: [`glmax.AbstractStdErrEstimator`][]; concrete strategies
          call it only if needed.

        **Returns:**

        [`glmax.InferenceResult`][] with fields `(params, se, stat, p)`.
        """


class WaldTest(AbstractTest, strict=True):
    r"""Wald (z/t) coefficient hypothesis test.

    Computes per-coefficient test statistics
    $z_j = \hat{\beta}_j / \operatorname{SE}(\hat{\beta}_j)$ and two-sided
    p-values. Here $\hat{\beta}_j$ is the fitted coefficient for term $j$ and
    $\operatorname{SE}(\hat{\beta}_j)$ is its estimated standard error. Uses a
    $t_{n-p}$ reference distribution for Gaussian models and
    $\mathcal{N}(0, 1)$ for all others, where $n$ is the number of
    observations and $p$ is the number of coefficients.

    Standard errors are obtained from the injected
    [`glmax.AbstractStdErrEstimator`][].
    """

    def __call__(
        self,
        fitted: FittedGLM,
        stderr: AbstractStdErrEstimator,
    ) -> InferenceResult:
        if not isinstance(fitted, FittedGLM):
            raise TypeError("WaldTest(...) expects `fitted` to be a FittedGLM instance.")
        if not isinstance(stderr, AbstractStdErrEstimator):
            raise TypeError("WaldTest(...) expects `stderr` to be an AbstractStdErrEstimator instance.")

        fit_result = fitted.result

        beta = fit_result.beta
        covariance = stderr(fitted)
        se = jnp.sqrt(jnp.diag(covariance))
        stat = beta / se
        df = int(fit_result.eta.shape[0] - beta.shape[0])
        p = _wald_test(stat, df, fitted.model)

        return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)


class ScoreTest(AbstractTest, strict=True):
    r"""Per-coefficient MLE-point score-style statistic built from fit artifacts.

    Computes the per-coefficient score statistic directly from `score_residual`,
    `glm_wt`, and the Fisher-information diagonal without calling `stderr`.
    The resulting statistic is normalised with a standard normal reference
    distribution. This is an MLE-point diagnostic, not a restricted-model Rao
    score test.

    `se` is set to NaN because no standard error carrier is exposed. Callers
    relying on [`glmax.InferenceResult`][].`se` downstream must handle NaN
    when using this inferrer.
    """

    def __call__(
        self,
        fitted: FittedGLM,
        stderr: AbstractStdErrEstimator,
    ) -> InferenceResult:
        if not isinstance(fitted, FittedGLM):
            raise TypeError("ScoreTest(...) expects `fitted` to be a FittedGLM instance.")
        if not isinstance(stderr, AbstractStdErrEstimator):
            raise TypeError("ScoreTest(...) expects `stderr` to be an AbstractStdErrEstimator instance.")

        fit_result = fitted.result

        X = fit_result.X
        glm_wt = fit_result.glm_wt
        score_residual = fit_result.score_residual
        beta = jnp.asarray(fit_result.params.beta)
        phi = _validated_fitted_dispersion(fitted)

        numerator = X.T @ (glm_wt * score_residual)
        fisher_diag = jnp.sum(X * (glm_wt[:, jnp.newaxis] * X), axis=0)
        fisher_diag = eqx.error_if(
            fisher_diag,
            ~jnp.all(jnp.isfinite(fisher_diag)) | ~jnp.all(fisher_diag > 0.0),
            "ScoreTest requires the Fisher information diagonal to be finite and > 0.",
        )

        stat = numerator / jnp.sqrt(phi * fisher_diag)
        p = 2.0 * norm.sf(jnp.abs(stat))
        se = jnp.full(beta.shape, jnp.nan)

        return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)


def _wald_test(statistic: ArrayLike, df: int, model: GLM) -> Array:
    r"""Two-sided Wald test p-values.

    Uses a $t_{df}$ distribution for Gaussian families and
    $\mathcal{N}(0, 1)$ for all others. Here `df` is the residual degrees of
    freedom. The statistic vector is typically
    $\hat{\beta} / \operatorname{SE}(\hat{\beta})$, where $\hat{\beta}$ is
    the fitted coefficient vector.

    **Arguments:**

    - `statistic`: test statistic vector, shape `(p,)`.
    - `df`: residual degrees of freedom $n - p$.
    - `model`: fitted [`glmax.GLM`][] instance.

    **Returns:**

    Two-sided p-values, shape `(p,)`.
    """
    if isinstance(model.family, Gaussian):
        return 2 * t_cdf(-jnp.abs(statistic), df)
    return 2 * norm.sf(jnp.abs(statistic))
