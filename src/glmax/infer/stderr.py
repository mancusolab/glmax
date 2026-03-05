from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jax.numpy import linalg as jnpla
from jaxtyping import ArrayLike, ScalarLike

from ..family.dist import ExponentialFamily


class AbstractStdErrEstimator(eqx.Module, strict=True):
    """
    This class defines an interface for computing **standard errors** in statistical models,
    it serves as a base class for implementations like `FisherInfoError` and `HuberError`.

    Subclasses must implement the `__call__` method, which computes standard errors
    of choice

    Parameters:
    ----------
    family : ExponentialFamily
        The Generalized Linear Model (GLM) distribution
    :param X : ArrayLike
        The **covariate data matrix** of shape (N, P), where:
        - **N** is the number of observations
        - **P** is the number of predictors
    :param y : ArrayLike
        The outcome vector of shape (N, 1), representing the observed response variable.
    :param eta : ArrayLike
        The linear predictor component
    :param mu : ArrayLike
        The fitted mean response values (expected values of `y`).
    :param weight : ArrayLike
        A weighting vector for each individual observation, used in variance estimation.
    :param disp : ScalarLike, optional (default=0.0)
        The dispersion parameter (specific to models like the **Negative Binomial**).

    Returns:
    -------
    Array
        The computed **standard errors** for the given model parameters.
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
        disp: ScalarLike = 0.0,
    ) -> Array:
        """calculate standard errors for SNP

        :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
        :param X: covariate data matrix (nxp)
        :param y: outcome vector (nx1)
        :param eta: linear component eta
        :param mu: fitted mean
        :param weight: weight for each individual
        :param disp: canonical dispersion parameter in NB model
        """
        pass


class FisherInfoError(AbstractStdErrEstimator, strict=True):
    """
    This class is an estimator for the standard errors of parameter estimates
    in models from the Exponential Family. Given a weighted design matrix `X`
    and other necessary parameters, it calculates the **asymptotic covariance matrix**
    by inverting the Fisher information matrix.

    Parameters:
    ----------
    :family : ExponentialFamily
        The exponential family distribution associated with the model.
    :param X : ArrayLike
        The design matrix (features) used in the estimation.
    :param y : ArrayLike
        The observed response variable.
    :param eta : ArrayLike
        The natural parameter of the exponential family distribution.
    :param mu : ArrayLike
        The mean response values (expected values of `y`).
    :param weight : ArrayLike
        A vector of weights applied to the observations.
    :param disp : ScalarLike, optional (default=0.0)
        A regularization parameter.

    Returns:
    -------
    Array
        The asymptotic covariance matrix, computed as the inverse of the Fisher
        information matrix.
    """

    def __call__(
        self,
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        weight: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> Array:
        del eta, mu, disp
        infor = (X * weight[:, jnp.newaxis]).T @ X
        asmpt_cov = jnpla.inv(infor)

        return asmpt_cov


class HuberError(AbstractStdErrEstimator, strict=True):
    """
    This class estimates the standard errors of parameter estimates in models
    from the Exponential Family using **Huber’s robust standard error approach**.
    It calculates a **sandwich estimator** for the asymptotic covariance matrix,
    accounting for heteroskedasticity or model misspecification.

    Parameters:
    ----------
    family : ExponentialFamily
        The exponential family distribution associated with the model.
    :param X : ArrayLike
        The design matrix (features) used in the estimation.
    :param y : ArrayLike
        The observed response variable.
    :param eta : ArrayLike
        The natural parameter of the exponential family distribution.
    :param mu : ArrayLike
        The mean response values (expected values of `y`).
    :param weight : ArrayLike
        A vector of weights applied to the observations.
    :param disp : ScalarLike, optional (default=0.0)
        A regularization parameter.

    Returns:
    -------
    Array
        The robust covariance matrix, computed using Huber’s sandwich formula.
    """

    def __call__(
        self,
        family: ExponentialFamily,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        mu: ArrayLike,
        weight: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> Array:
        # note: this scaler will cancel out in robust_cov
        phi = family.scale(X, y, mu)
        gprime = family.glink.deriv(mu)

        # calculate observed hessian
        W = 1 / phi * (family._hlink_score(eta, disp) / gprime - family._hlink_hess(eta, disp) * (y - mu))
        hess_inv = jnpla.inv(-(X * W).T @ X)

        score_no_x = (y - mu) / (family.variance(mu, disp) * gprime * phi)
        Bs = (X * (score_no_x**2)).T @ X
        robust_cov = hess_inv @ Bs @ hess_inv

        return robust_cov
