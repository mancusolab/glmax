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
    """
    Base protocol for covariance estimators used by `_infer(fitted, stderr=...)`.
    """

    @abstractmethod
    def __call__(self, fitted: "FittedGLM") -> Array:
        """Return a covariance matrix for `fitted.result.params.beta`."""


class FisherInfoError(AbstractStdErrEstimator, strict=True):
    """
    Covariance estimator based on Fisher information reconstructed from fit artifacts.
    """

    def __call__(self, fitted: "FittedGLM") -> Array:
        r"""Compute $\hat{\mathrm{Cov}}(\hat\beta) = \phi \cdot \mathcal{I}(\hat\beta)^{-1}$.

        **Arguments:**

        - `fitted`: validated `FittedGLM` containing the model and fit artifacts.

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
    """
    Sandwich covariance estimator over fit artifacts.
    """

    def __call__(self, fitted: "FittedGLM") -> Array:
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
