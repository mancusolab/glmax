# pattern: Functional Core

"""Internal standard-error estimators used by GLM fit/_infer kernels."""

import math

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jax.numpy import linalg as jnpla

from .._fit import FittedGLM


def _validated_fitted_dispersion(fitted: FittedGLM) -> Array:
    """Return canonical fitted dispersion after finite/positive validation."""
    phi = jnp.asarray(fitted.params.disp)
    try:
        phi_scalar = float(phi)
    except TypeError:
        phi_scalar = None

    if phi_scalar is not None:
        if not math.isfinite(phi_scalar) or phi_scalar <= 0.0:
            raise ValueError("Inference requires fitted.params.disp to be finite and > 0.")
        return phi

    return eqx.error_if(
        phi,
        ~jnp.isfinite(phi) | (phi <= 0.0),
        "Inference requires fitted.params.disp to be finite and > 0.",
    )


class AbstractStdErrEstimator(eqx.Module, strict=True):
    r"""Abstract base for covariance estimators used by `infer(fitted, stderr=...)`.

    Subclasses implement `covariance` to return a `(p, p)` covariance matrix for
    $\hat{\beta}$. The matrix is consumed by [`glmax.AbstractTest`][]
    strategies to compute standard errors and test statistics.
    """

    @abstractmethod
    def covariance(self, fitted: FittedGLM) -> Array:
        r"""Estimate the covariance matrix for `fitted.result.params.beta`.

        The returned matrix estimates
        $\widehat{\operatorname{Cov}}(\hat{\beta})$, where $\hat{\beta}$ is
        the fitted coefficient vector.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.

        **Returns:**

        Covariance matrix
        $\widehat{\operatorname{Cov}}(\hat{\beta})$, shape `(p, p)`.
        """


class FisherInfoError(AbstractStdErrEstimator, strict=True):
    r"""Fisher-information covariance estimator.

    Reconstructs the expected Fisher information from fit artifacts and inverts it.
    Default estimator used by `WaldTest`.
    """

    def covariance(self, fitted: FittedGLM) -> Array:
        r"""Compute a Fisher-information covariance estimate.

        This computes
        $\widehat{\operatorname{Cov}}(\hat{\beta}) = \hat{\phi}\,
        \mathcal{I}(\hat{\beta})^{-1}$, where $\hat{\phi}$ is the fitted
        dispersion and $\mathcal{I}(\hat{\beta})$ is the expected Fisher
        information evaluated at the fitted coefficients.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.

        **Returns:**

        Covariance matrix, shape `(p, p)`.
        """
        fit_result = fitted.result
        phi = _validated_fitted_dispersion(fitted)
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

    def covariance(self, fitted: FittedGLM) -> Array:
        r"""Compute the sandwich covariance matrix.

        This computes the sandwich estimator
        $\widehat{\operatorname{Cov}}(\hat{\beta}) = B M B$, where
        $B = \hat{\phi}\,\mathcal{I}^{-1}$ is the bread matrix and
        $M = X^\top \operatorname{diag}(\hat{s}_i^2) X$ is the meat matrix.
        Here $\hat{s}_i$ denotes the score contribution for observation $i$.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.

        **Returns:**

        Sandwich covariance matrix, shape `(p, p)`.
        """
        fit_result = fitted.result
        X = fit_result.X
        phi = _validated_fitted_dispersion(fitted)
        w_pure = fit_result.glm_wt * phi
        bread = phi * jnpla.inv((X * w_pure[:, jnp.newaxis]).T @ X)

        # score_i = x_i * ((y_i - mu_i) / (V(mu_i) g'(mu_i) phi))
        # and glm_wt = 1 / (V(mu) g'(mu)^2), so the scalar score term is
        # glm_wt * score_residual / phi with score_residual = (y - mu) g'(mu).
        score_no_x = fit_result.glm_wt * fit_result.score_residual / phi
        meat = (X * (score_no_x**2)[:, jnp.newaxis]).T @ X
        return bread @ meat @ bread
