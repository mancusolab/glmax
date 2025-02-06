from abc import abstractmethod
from typing import ClassVar, List, Tuple, Type

import numpy as np

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.stats as jaxstats

from jaxtyping import Array, ArrayLike, ScalarLike

from .links import Identity, Link, Log, Power


class ExponentialFamily(eqx.Module):
    """
    Define parent class for exponential family distribution (One parameter EF for now).
    Provide all required link function relevant to generalized linear model (GLM).
    GLM: g(mu) = X @ b, where mu = E(Y|X)
    : hlink : h(X @ b) = b'-1 (g^-1(X @ b)) = theta, default is canonical link which returns identity function.
    : hlink_der : derivative of hlink function
    : glink : g(mu) = X @ b, canonical link is g = b'-1, allows user to provide other link function.
    : glink_inv : inverse of glink, where g^-1(X @ b) = mu
    : glink_der : derivative of glink
    : log_prob : log joint density of all observations
    """

    glink: Link
    _links: ClassVar[List[Type[Link]]]

    def __check_init__(self):
        if not any([isinstance(self.glink, link) for link in self._links]):
            raise ValueError(f"Link {self.glink} is invalid for Family {self}")

    @abstractmethod
    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        # phi is the dispersion parameter
        pass

    @abstractmethod
    def negloglikelihood(self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ScalarLike) -> Array:
        pass

    @abstractmethod
    def variance(self, mu: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        pass

    def score(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        """
        For canonical link, this is X^t (y - mu)/phi, phi is the self.scale
        """
        return -X.T @ (y - mu) / self.scale(X, y, mu)

    def random_gen(self, mu: ArrayLike, scale: ScalarLike = 1.0, alpha: ScalarLike = 0.0) -> Array:
        pass

    def calc_weight(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Tuple[Array, Array, Array]:
        """
        weight for each observation in IRLS
        weight_i = 1 / (V(mu_i) * phi * g'(mu_i)**2)
        this is part of the Information matrix
        """
        mu_k = self.glink.inverse(eta)
        g_deriv_k = self.glink.deriv(mu_k)
        phi = self.scale(X, y, mu_k)
        weight_k = 1.0 / (phi * self.variance(mu_k, alpha) * g_deriv_k**2)

        return mu_k, g_deriv_k, weight_k

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + y.mean()) / 2)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.01,
        step_size: ScalarLike = 1.0,
    ) -> Array:
        return jnp.asarray(0.0)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.01,
        step_size: ScalarLike = 1.0,
        tol: ScalarLike = 1e-3,
        max_iter: int = 1000,
        offset_eta: ScalarLike = 0.0,
    ) -> Array:
        return jnp.asarray(0.0)

    def _hlink(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        """
        If canonical link, then this is identity function
        """
        return jnp.asarray(eta)

    def _hlink_score(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        """
        If canonical link, then this is identity function
        """
        return jnp.ones_like(eta)

    def _hlink_hess(self, eta: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return jnp.zeros_like(eta)


class Gaussian(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2),
    we can treat normal distribution as one-parameter EF
    """

    glink: Link = Identity()
    _links: ClassVar[List[Type[Link]]] = [Identity, Log, Power]

    def random_gen(self, mu: ArrayLike, scale: ScalarLike = 1.0, alpha: ScalarLike = 0.0) -> Array:
        y = np.random.normal(mu, scale)
        return y

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        resid = jnp.sum(jnp.square(mu - y))
        df = y.shape[0] - X.shape[1]
        phi = resid / df
        return phi

    def negloglikelihood(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.0,
    ) -> Array:
        mu = self.glink.inverse(eta)
        phi = self.scale(X, y, mu)
        logprob = jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(phi)))
        return -logprob

    def variance(self, mu: ArrayLike, alpha: ScalarLike = 0.0) -> Array:
        return jnp.ones_like(mu)
