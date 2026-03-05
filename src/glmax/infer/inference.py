# pattern: Functional Core

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jax.numpy import linalg as jnpla
from jax.scipy.stats import norm
from jaxtyping import ArrayLike, ScalarLike

from ..family.dist import ExponentialFamily, Gaussian
from ..family.utils import t_cdf


def wald_test(statistic: ArrayLike, df: int, family: ExponentialFamily) -> Array:
    r"""Compute Wald-test p-values for fitted coefficients.

    **Arguments:**

    - `statistic`: Wald statistics, typically `beta / se`.
    - `df`: Degrees of freedom used for Gaussian-family t-reference.
    - `family`: Exponential-family model determining p-value reference distribution.

    **Returns:**

    - P-value array aligned with `statistic`.

    **Failure Modes:**

    - Assumes caller provides finite statistics and valid degrees of freedom from
      validated fit-boundary inputs.
    """
    if isinstance(family, Gaussian):
        return 2 * t_cdf(-abs(statistic), df)
    return 2 * norm.sf(abs(statistic))


class AbstractStdErrEstimator(eqx.Module, strict=True):
    r"""Abstract contract for covariance/standard-error estimators.

    **Arguments:**

    - Implementations accept validated GLM boundary outputs (`X`, `y`, `eta`, `mu`,
      `weight`, and `alpha`) and family metadata.

    **Returns:**

    - Covariance matrix used to derive standard errors.

    **Failure Modes:**

    - Implementations may raise backend linear-algebra errors for singular or
      ill-conditioned covariance operators.
    """

    @abstractmethod
    def __call__(
        self,
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        weight: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        r"""Compute covariance estimate for one GLM fit.

        **Arguments:**

        - `family`: Exponential-family model metadata.
        - `X`: Covariate matrix with shape `(n, p)`.
        - `y`: Response vector with shape `(n,)`.
        - `eta`: Linear predictor vector.
        - `mu`: Mean-response vector.
        - `weight`: Per-sample working weights.
        - `alpha`: Dispersion parameter.

        **Returns:**

        - Covariance matrix with shape `(p, p)`.

        **Failure Modes:**

        - Implementations may raise linear-algebra errors if covariance operators
          are singular or invalid.
        """
        pass


class FisherInfoError(AbstractStdErrEstimator, strict=True):
    r"""Fisher-information covariance estimator.

    **Arguments:**

    - Consumes validated GLM fit intermediates and working weights.

    **Returns:**

    - Inverse Fisher-information covariance estimate.

    **Failure Modes:**

    - Matrix inversion may fail for singular information operators.
    """

    def __call__(
        self,
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        weight: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        del eta, mu, alpha
        infor = (X * weight[:, jnp.newaxis]).T @ X
        asmpt_cov = jnpla.inv(infor)

        return asmpt_cov


class HuberError(AbstractStdErrEstimator, strict=True):
    r"""Huber sandwich covariance estimator.

    **Arguments:**

    - Consumes validated GLM fit intermediates and working weights.

    **Returns:**

    - Robust sandwich covariance estimate.

    **Failure Modes:**

    - Hessian inversion may fail for singular or numerically unstable operators.
    """

    def __call__(
        self,
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        weight: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        # note: this scaler will cancel out in robust_cov
        phi = family.scale(X, y, mu)
        gprime = family.glink.deriv(mu)

        # calculate observed hessian
        W = 1 / phi * (family._hlink_score(eta, alpha) / gprime - family._hlink_hess(eta, alpha) * (y - mu))
        hess_inv = jnpla.inv(-(X * W).T @ X)

        score_no_x = (y - mu) / (family.variance(mu, alpha) * gprime * phi)
        Bs = (X * (score_no_x**2)).T @ X
        robust_cov = hess_inv @ Bs @ hess_inv

        return robust_cov
