# pattern: Functional Core
"""Link functions g(mu) = eta for GLM families."""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jspec

from jaxtyping import Array, ArrayLike, Scalar

from .utils import _clipped_expit, _grad_per_sample


class AbstractLink(eqx.Module):
    r"""Abstract base for GLM link functions $g: \mu \mapsto \eta$.

    A link function connects the mean parameter $\mu = \mathrm{E}(Y | X)$
    to the linear predictor $\eta = X\beta$ via $\eta = g(\mu)$.
    """

    @abstractmethod
    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute $g(\mu) = \eta$.

        **Arguments:**

        - `mu`: mean parameter, shape `(n,)`, entries in the family's support.

        **Returns:**

        Linear predictor $\eta$, shape `(n,)`.
        """

    @abstractmethod
    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}(\eta) = \mu$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Mean parameter $\mu$, shape `(n,)`.
        """

    @abstractmethod
    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu)$.

        **Arguments:**

        - `mu`: mean parameter, shape `(n,)`.

        **Returns:**

        Link derivative, shape `(n,)`.
        """

    @abstractmethod
    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $(g^{-1})'(\eta)$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Inverse-link derivative, shape `(n,)`.
        """


class PowerLink(AbstractLink):
    r"""Power link $g(\mu) = \mu^p$."""

    power: Scalar

    def __init__(self, power: float = 1.0) -> None:
        r"""**Arguments:**
        - `power`: exponent $p$. Default `1.0` (identity link).
        """
        self.power = jnp.asarray(power)

    def __check_init__(self):
        if self.power == 0:
            raise ValueError("PowerLink: power=0 is degenerate (inverse is undefined)")

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute $g(\mu) = \mu^p$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $> 0$.

        **Returns:**

        $\mu^p$, shape `(n,)`.
        """
        return jnp.power(mu, self.power)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}(\eta) = \eta^{1/p}$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $\eta^{1/p}$, shape `(n,)`.
        """
        return jnp.power(eta, 1.0 / self.power)

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu) = p \mu^{p-1}$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $p \mu^{p-1}$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $(g^{-1})'(\eta) = \eta^{1/p-1}/p$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $\eta^{1/p-1}/p$, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class IdentityLink(AbstractLink):
    r"""Identity link $g(\mu) = \mu$. The canonical link for the Gaussian family."""

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute $g(\mu) = \mu$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $\mu$, shape `(n,)`.
        """
        return mu

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}(\eta) = \eta$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $\eta$, shape `(n,)`.
        """
        return eta

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu) = 1$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        Ones, shape `(n,)`.
        """
        return jnp.ones_like(mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $(g^{-1})'(\eta) = 1$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Ones, shape `(n,)`.
        """
        return jnp.ones_like(eta)


class LogitLink(AbstractLink):
    r"""Logit link $g(\mu) = \log(\mu / (1 - \mu))$.

    The canonical link for the Binomial family.
    """

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute $g(\mu) = \log(\mu / (1 - \mu))$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries in $(0, 1)$.

        **Returns:**

        Log-odds, shape `(n,)`.
        """
        return jspec.logit(mu)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}(\eta) = \sigma(\eta)$, clipped to $(0, 1)$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Clipped sigmoid, shape `(n,)`.
        """
        return _clipped_expit(eta)

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu) = 1 / (\mu(1 - \mu))$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $1/(\mu(1-\mu))$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $(g^{-1})'(\eta) = \sigma(\eta)(1 - \sigma(\eta))$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Sigmoid derivative, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class InverseLink(AbstractLink):
    r"""Reciprocal link $g(\mu) = 1/\mu$. The canonical link for Gamma models."""

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute $g(\mu) = 1/\mu$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $> 0$.

        **Returns:**

        $1/\mu$, shape `(n,)`.
        """
        return jnp.reciprocal(mu)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}(\eta) = 1/\eta$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`, entries $\neq 0$.

        **Returns:**

        $1/\eta$, shape `(n,)`.
        """
        eta = jnp.asarray(eta)
        # Gamma link: eta = 1/mu > 0; clip to tiny to prevent inf at eta=0.
        # Use result_type(eta) so float32 inputs stay float32; the project
        # default is float64 (jax_enable_x64 set at import), but this guard
        # must not upcast via a hardcoded jnp.finfo(jnp.float64).tiny.
        tiny = jnp.finfo(jnp.result_type(eta)).tiny
        eta = jnp.where(jnp.abs(eta) < tiny, tiny, eta)
        return jnp.reciprocal(eta)

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu) = -1/\mu^2$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $-1/\mu^2$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $(g^{-1})'(\eta) = -1/\eta^2$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $-1/\eta^2$, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class LogLink(AbstractLink):
    r"""Log link $g(\mu) = \log(\mu)$. The canonical link for the Poisson family."""

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute $g(\mu) = \log(\mu)$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $> 0$.

        **Returns:**

        $\log(\mu)$, shape `(n,)`.
        """
        return jnp.log(mu)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}(\eta) = e^\eta$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $e^\eta$, shape `(n,)`.
        """
        return jnp.exp(eta)

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu) = 1/\mu$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $1/\mu$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $(g^{-1})'(\eta) = e^\eta$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $e^\eta$, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class NBLink(AbstractLink):
    r"""Negative-binomial link $g(\mu) = \log(\alpha\mu / (1 + \alpha\mu))$."""

    alpha: Scalar

    def __init__(self, alpha: float = 1.0) -> None:
        r"""**Arguments:**
        - `alpha`: overdispersion parameter $\alpha > 0$. Default `1.0`.
        """
        self.alpha = jnp.asarray(alpha)

    def __call__(self, mu: ArrayLike) -> Array:
        r"""Compute $g(\mu) = \log(\alpha\mu / (1 + \alpha\mu))$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $> 0$.

        **Returns:**

        NB log-link value, shape `(n,)`.
        """
        # log(x) - log1p(x) is numerically more stable than log(x/(x+1))
        # at large mu*alpha, where x/(x+1) rounds to 1 and log returns 0.
        mu_alpha = jnp.asarray(mu) * self.alpha
        return jnp.log(mu_alpha) - jnp.log1p(mu_alpha)

    def inverse(self, eta: ArrayLike) -> Array:
        r"""Compute $g^{-1}(\eta) = 1 / (\alpha \operatorname{expm1}(-\eta))$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`, entries $< 0$.

        **Returns:**

        Mean $\mu$, shape `(n,)`.
        """
        return 1.0 / (self.alpha * jnp.expm1(-eta))

    def deriv(self, mu: ArrayLike) -> Array:
        r"""Compute $g'(\mu) = 1 / (\mu(1 + \alpha\mu))$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        NB link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: ArrayLike) -> Array:
        r"""Compute $(g^{-1})'(\eta)$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Inverse-link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)
