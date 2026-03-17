# pattern: Functional Core

"""Internal standard-error estimators used by GLM fit/_infer kernels."""

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jax.numpy import linalg as jnpla


if TYPE_CHECKING:
    from .. import FittedGLM


class AbstractStdErrEstimator(eqx.Module, strict=True):
    r"""Abstract base for covariance estimators used by `infer(fitted, stderr=...)`.

    Subclasses implement `__call__` to return a `(p, p)` covariance matrix for
    $\hat\beta$. The matrix is consumed by `AbstractTest` strategies to compute
    standard errors and test statistics.
    """

    @abstractmethod
    def __call__(self, fitted: "FittedGLM") -> Array:
        r"""Estimate the covariance matrix for `fitted.result.params.beta`.

        **Arguments:**

        - `fitted`: `FittedGLM` from `fit(...)`.

        **Returns:**

        Covariance matrix $\hat{\mathrm{Cov}}(\hat\beta)$, shape `(p, p)`.
        """


class FisherInfoError(AbstractStdErrEstimator, strict=True):
    r"""Fisher-information covariance estimator.

    Reconstructs the expected Fisher information from fit artifacts and inverts it.
    Default estimator used by `WaldTest`.
    """

    def __call__(self, fitted: "FittedGLM") -> Array:
        r"""Compute $\hat{\mathrm{Cov}}(\hat\beta) = \hat\phi \cdot \mathcal{I}(\hat\beta)^{-1}$.

        **Arguments:**

        - `fitted`: `FittedGLM` from `fit(...)`.

        **Returns:**

        Covariance matrix, shape `(p, p)`.
        """
        model = fitted.model
        fit_result = fitted.result
        phi = jnp.asarray(model.scale(fit_result.X, fit_result.y, fit_result.mu))
        w_pure = fit_result.glm_wt * phi
        information = (fit_result.X * w_pure[:, jnp.newaxis]).T @ fit_result.X
        return phi * jnpla.inv(information)


class HuberError(AbstractStdErrEstimator, strict=True):
    r"""Huber-White sandwich covariance estimator.

    Computes the heteroskedasticity-robust "meat-bread" sandwich estimator
    $\hat{\mathrm{Cov}}(\hat\beta) = B \, M \, B$ where $B = \hat\phi \, \mathcal{I}^{-1}$
    and $M = X^\top \mathrm{diag}(\hat{s}_i^2) X$ with per-observation score contributions
    $\hat{s}_i = w_i r_i / \hat\phi$.
    """

    def __call__(self, fitted: "FittedGLM") -> Array:
        r"""Compute the sandwich covariance matrix.

        **Arguments:**

        - `fitted`: `FittedGLM` from `fit(...)`.

        **Returns:**

        Sandwich covariance matrix, shape `(p, p)`.
        """
        model = fitted.model
        fit_result = fitted.result
        X = fit_result.X
        phi = jnp.asarray(model.scale(X, fit_result.y, fit_result.mu))
        w_pure = fit_result.glm_wt * phi
        bread = phi * jnpla.inv((X * w_pure[:, jnp.newaxis]).T @ X)

        # score_i = x_i * ((y_i - mu_i) / (V(mu_i) g'(mu_i) phi))
        # and glm_wt = 1 / (V(mu) g'(mu)^2), so the scalar score term is
        # glm_wt * score_residual / phi with score_residual = (y - mu) g'(mu).
        score_no_x = fit_result.glm_wt * fit_result.score_residual / phi
        meat = (X * (score_no_x**2)[:, jnp.newaxis]).T @ X
        return bread @ meat @ bread
