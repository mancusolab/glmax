# pattern: Functional Core

"""Inferrer strategies for the `_infer()` verb."""

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from .._fit import FittedGLM
from ..family import ExponentialFamily, Gaussian
from ..family.utils import t_cdf
from .stderr import AbstractStdErrEstimator
from .types import InferenceResult


__all__ = ["AbstractTest", "WaldTest", "ScoreTest"]


class AbstractTest(eqx.Module, strict=True):
    """Base class for inference strategies used by `_infer(fitted, inferrer=...)`."""

    @abstractmethod
    def __call__(
        self,
        fitted: "FittedGLM",
        stderr: AbstractStdErrEstimator,
    ) -> InferenceResult:
        """Compute inferential summaries from a fitted GLM.

        **Arguments:**

        - `fitted`: validated `FittedGLM` from `fit()`.
        - `stderr`: standard-error estimator; concrete inferrers call it only if needed.

        **Returns:**

        `InferenceResult` with `(params, se, stat, p)`.
        """


class WaldTest(AbstractTest, strict=True):
    """Wald (z/t) hypothesis test."""

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
        p = _wald_test(stat, df, fitted.model.family)

        return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)


class ScoreTest(AbstractTest, strict=True):
    """Per-coefficient MLE-point score-style statistic built from fit artifacts.

    Computes the per-coefficient score statistic directly from `score_residual`,
    `glm_wt`, and the Fisher-information diagonal without calling `stderr`.
    The resulting statistic is normalised with a standard normal reference
    distribution. This is an MLE-point diagnostic, not a restricted-model Rao
    score test.

    `se` is set to NaN because no standard error carrier is exposed. Callers
    relying on `InferenceResult.se` downstream must handle NaN when using this
    inferrer.
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
        y = fit_result.y
        mu = fit_result.mu
        glm_wt = fit_result.glm_wt
        score_residual = fit_result.score_residual
        beta = jnp.asarray(fit_result.params.beta)
        phi = jnp.asarray(fitted.model.family.scale(X, y, mu))
        if not bool(jnp.isfinite(phi)) or float(phi) <= 0.0:
            raise ValueError("ScoreTest requires family.scale(X, y, mu) to be finite and > 0.")

        numerator = X.T @ (glm_wt * score_residual)
        fisher_diag = jnp.sum(X * (glm_wt[:, jnp.newaxis] * X), axis=0)
        if not bool(jnp.all(jnp.isfinite(fisher_diag))) or not bool(jnp.all(fisher_diag > 0.0)):
            raise ValueError("ScoreTest requires the Fisher information diagonal to be finite and > 0.")

        stat = numerator / jnp.sqrt(phi * fisher_diag)
        p = 2.0 * norm.sf(jnp.abs(stat))
        se = jnp.full(beta.shape, jnp.nan)

        return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)


def _wald_test(statistic: ArrayLike, df: int, family: ExponentialFamily) -> Array:
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
