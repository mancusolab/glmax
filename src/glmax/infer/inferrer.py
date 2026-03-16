# pattern: Functional Core

"""Inferrer strategies for the `infer()` verb."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from jax.scipy.stats import norm

from ..fit import (
    _matches_fit_result_shape,
    _matches_fitted_glm_shape,
    validate_fit_result,
)
from .inference import InferenceResult, wald_test
from .stderr import AbstractStdErrEstimator


if TYPE_CHECKING:
    from ..fit import FittedGLM


__all__ = ["AbstractInferrer", "WaldInferrer", "ScoreInferrer", "DEFAULT_INFERRER"]


class AbstractInferrer(eqx.Module, strict=True):
    """Base class for inference strategies used by `infer(fitted, inferrer=...)`."""

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


class WaldInferrer(AbstractInferrer, strict=True):
    """Wald (z/t) hypothesis test."""

    def __call__(
        self,
        fitted: "FittedGLM",
        stderr: AbstractStdErrEstimator,
    ) -> InferenceResult:
        if not _matches_fitted_glm_shape(fitted):
            raise TypeError("WaldInferrer expects `fitted` to be a FittedGLM instance.")
        if not isinstance(stderr, AbstractStdErrEstimator):
            raise TypeError("WaldInferrer expects `stderr` to be an AbstractStdErrEstimator instance.")

        fit_result = fitted.result
        if not _matches_fit_result_shape(fit_result):
            raise TypeError("WaldInferrer expects `fitted.result` to be a FitResult instance.")
        validate_fit_result(fit_result)

        beta = jnp.asarray(fit_result.params.beta)
        covariance = jnp.asarray(stderr(fitted))
        se = jnp.sqrt(jnp.diag(covariance))
        stat = beta / se
        df = int(fit_result.eta.shape[0] - beta.shape[0])
        p = wald_test(stat, df, fitted.model.family)

        return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)


class ScoreInferrer(AbstractInferrer, strict=True):
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
        fitted: "FittedGLM",
        stderr: AbstractStdErrEstimator,
    ) -> InferenceResult:
        del stderr
        if not _matches_fitted_glm_shape(fitted):
            raise TypeError("ScoreInferrer expects `fitted` to be a FittedGLM instance.")

        fit_result = fitted.result
        validate_fit_result(fit_result)

        X = fit_result.X
        y = fit_result.y
        mu = fit_result.mu
        glm_wt = fit_result.glm_wt
        score_residual = fit_result.score_residual
        beta = jnp.asarray(fit_result.params.beta)
        phi = jnp.asarray(fitted.model.family.scale(X, y, mu))
        if not bool(jnp.isfinite(phi)) or float(phi) <= 0.0:
            raise ValueError("ScoreInferrer requires family.scale(X, y, mu) to be finite and > 0.")
        numerator = X.T @ (glm_wt * score_residual)
        fisher_diag = jnp.sum(X * (glm_wt[:, jnp.newaxis] * X), axis=0)
        if not bool(jnp.all(jnp.isfinite(fisher_diag))) or not bool(jnp.all(fisher_diag > 0.0)):
            raise ValueError("ScoreInferrer requires the Fisher information diagonal to be finite and > 0.")
        stat = numerator / jnp.sqrt(phi * fisher_diag)
        p = 2.0 * norm.sf(jnp.abs(stat))
        se = jnp.full(beta.shape, jnp.nan)

        return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)


DEFAULT_INFERRER: AbstractInferrer = WaldInferrer()
