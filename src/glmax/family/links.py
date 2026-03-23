# pattern: Functional Core
"""Link functions g(mu) = eta for GLM families."""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jspec

from jaxtyping import Array, Scalar

from .utils import _clipped_expit, _grad_per_sample


class AbstractLink(eqx.Module):
    r"""Abstract base for GLM link functions $g: \mu \mapsto \eta$.

    A link function connects the mean parameter $\mu = \mathrm{E}(Y | X)$
    to the linear predictor $\eta = X \beta$ via $\eta = g(\mu)$, where
    $X$ is the design matrix and $\beta$ is the coefficient vector.

    Every concrete link exposes four related maps:
    $g(\mu)$, its derivative $g'(\mu)$, the inverse link $g^{-1}(\eta)$, and
    the inverse-link derivative $(g^{-1})'(\eta)$.
    """

    @abstractmethod
    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \eta$.

        Here $\mu$ is the mean-response vector and $\eta$ is the linear
        predictor.

        **Arguments:**

        - `mu`: mean parameter, shape `(n,)`, entries in the family's support.

        **Returns:**

        Linear predictor $\eta$, shape `(n,)`.
        """

    @abstractmethod
    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = \mu$.

        Here $\eta$ is the linear predictor and $\mu$ is the implied mean
        response.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Mean parameter $\mu$, shape `(n,)`.
        """

    @abstractmethod
    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu)$.

        The derivative is evaluated elementwise with respect to the mean
        response $\mu$.

        **Arguments:**

        - `mu`: mean parameter, shape `(n,)`.

        **Returns:**

        Link derivative, shape `(n,)`.
        """

    @abstractmethod
    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta)$.

        The derivative is evaluated elementwise with respect to the linear
        predictor $\eta$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Inverse-link derivative, shape `(n,)`.
        """


class PowerLink(AbstractLink):
    r"""Power link $g(\mu) = \mu^p$.

    Here $\mu$ is the mean response and $p$ is the power exponent.
    The derivative is $g'(\mu) = p \mu^{p-1}$, the inverse link is
    $g^{-1}(\eta) = \eta^{1/p}$, and the inverse-link derivative is
    $(g^{-1})'(\eta) = \eta^{1/p - 1} / p$.
    """

    power: Scalar

    def __init__(self, power: float = 1.0) -> None:
        r"""Construct a power link.

        **Arguments:**

        - `power`: exponent $p$. Default `1.0` (identity link).
        """
        self.power = jnp.asarray(power)

    def __check_init__(self):
        if self.power == 0:
            raise ValueError("PowerLink: power=0 is degenerate (inverse is undefined)")

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \mu^p$.

        The symbol $p$ denotes the configured power exponent.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $> 0$.

        **Returns:**

        $\mu^p$, shape `(n,)`.
        """
        return jnp.power(mu, self.power)

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = \eta^{1/p}$.

        The symbol $p$ denotes the configured power exponent.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $\eta^{1/p}$, shape `(n,)`.
        """
        return jnp.power(eta, 1.0 / self.power)

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = p \mu^{p-1}$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $p \mu^{p-1}$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = \eta^{1/p-1}/p$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $\eta^{1/p-1}/p$, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class IdentityLink(AbstractLink):
    r"""Identity link $g(\mu) = \mu$.

    The derivative is $g'(\mu) = 1$, the inverse link is
    $g^{-1}(\eta) = \eta$, and the inverse-link derivative is
    $(g^{-1})'(\eta) = 1$. This is the canonical link for the Gaussian family.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \mu$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $\mu$, shape `(n,)`.
        """
        return mu

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = \eta$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $\eta$, shape `(n,)`.
        """
        return eta

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = 1$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        Ones, shape `(n,)`.
        """
        return jnp.ones_like(mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = 1$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Ones, shape `(n,)`.
        """
        return jnp.ones_like(eta)


class LogitLink(AbstractLink):
    r"""Logit link $g(\mu) = \log(\mu / (1 - \mu))$.

    The derivative is $g'(\mu) = 1 / (\mu (1 - \mu))$, the inverse link is
    $g^{-1}(\eta) = \sigma(\eta)$ where $\sigma$ is the logistic sigmoid, and
    the inverse-link derivative is
    $(g^{-1})'(\eta) = \sigma(\eta) (1 - \sigma(\eta))$.

    This is the canonical link for the Binomial family.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \log(\mu / (1 - \mu))$.

        Here $\mu \in (0, 1)$ is the Bernoulli success probability and
        $\eta$ is the log-odds.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries in $(0, 1)$.

        **Returns:**

        Log-odds, shape `(n,)`.
        """
        return jspec.logit(mu)

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = \sigma(\eta)$, clipped to $(0, 1)$.

        The symbol $\sigma$ denotes the logistic sigmoid function.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Clipped sigmoid, shape `(n,)`.
        """
        return _clipped_expit(eta)

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = 1 / (\mu(1 - \mu))$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $1/(\mu(1-\mu))$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = \sigma(\eta)(1 - \sigma(\eta))$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Sigmoid derivative, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class InverseLink(AbstractLink):
    r"""Reciprocal link $g(\mu) = 1 / \mu$.

    The derivative is $g'(\mu) = -1 / \mu^2$, the inverse link is
    $g^{-1}(\eta) = 1 / \eta$, and the inverse-link derivative is
    $(g^{-1})'(\eta) = -1 / \eta^2$.

    This is the canonical link for Gamma models.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = 1/\mu$.

        This link requires strictly positive means $\mu > 0$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $> 0$.

        **Returns:**

        $1/\mu$, shape `(n,)`.
        """
        return jnp.reciprocal(mu)

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = 1/\eta$.

        This inverse uses a tiny-value guard so values near $\eta = 0$ do
        not produce infinities.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`, entries $\neq 0$.

        **Returns:**

        $1/\eta$, shape `(n,)`.
        """
        # Gamma link: eta = 1/mu > 0; clip to tiny to prevent inf at eta=0.
        # Use result_type(eta) so float32 inputs stay float32; the project
        # default is float64 (jax_enable_x64 set at import), but this guard
        # must not upcast via a hardcoded jnp.finfo(jnp.float64).tiny.
        tiny = jnp.finfo(jnp.result_type(eta)).tiny
        eta = jnp.where(jnp.abs(eta) < tiny, tiny, eta)
        return jnp.reciprocal(eta)

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = -1/\mu^2$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $-1/\mu^2$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = -1/\eta^2$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $-1/\eta^2$, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class LogLink(AbstractLink):
    r"""Log link $g(\mu) = \log(\mu)$.

    The derivative is $g'(\mu) = 1 / \mu$, the inverse link is
    $g^{-1}(\eta) = e^\eta$, and the inverse-link derivative is
    $(g^{-1})'(\eta) = e^\eta$.

    This is the canonical link for the Poisson family.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \log(\mu)$.

        This link requires strictly positive means $\mu > 0$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $> 0$.

        **Returns:**

        $\log(\mu)$, shape `(n,)`.
        """
        return jnp.log(mu)

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = e^\eta$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $e^\eta$, shape `(n,)`.
        """
        return jnp.exp(eta)

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = 1/\mu$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $1/\mu$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = e^\eta$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $e^\eta$, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class ProbitLink(AbstractLink):
    r"""Probit link $g(\mu) = \Phi^{-1}(\mu)$.

    Here $\Phi^{-1}$ is the standard normal quantile function (inverse CDF).
    The inverse link is $g^{-1}(\eta) = \Phi(\eta)$, where $\Phi$ is the
    standard normal CDF. The derivative $g'(\mu)$ and inverse-link derivative
    $(g^{-1})'(\eta)$ are computed via autodiff.

    This is the second most common link for the Binomial family, widely used
    in bioassay and econometrics.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \Phi^{-1}(\mu)$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries in $(0, 1)$.

        **Returns:**

        Standard normal quantile $\Phi^{-1}(\mu)$, shape `(n,)`.
        """
        return jspec.ndtri(mu)

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = \Phi(\eta)$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Standard normal CDF $\Phi(\eta)$, shape `(n,)`.
        """
        return jspec.ndtr(eta)

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = 1 / \phi(\Phi^{-1}(\mu))$ via autodiff.

        Here $\phi$ is the standard normal PDF.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        Probit link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = \phi(\eta)$ via autodiff.

        Here $\phi$ is the standard normal PDF.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Standard normal PDF $\phi(\eta)$, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class CLogLogLink(AbstractLink):
    r"""Complementary log-log link $g(\mu) = \log(-\log(1 - \mu))$.

    The inverse link is $g^{-1}(\eta) = 1 - \exp(-\exp(\eta))$.
    The derivative $g'(\mu)$ and inverse-link derivative $(g^{-1})'(\eta)$
    are computed via autodiff.

    This is the canonical link for interval-censored survival models and
    log-Weibull regression. It is asymmetric: the probability approaches 0
    slower than it approaches 1.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \log(-\log(1 - \mu))$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries in $(0, 1)$.

        **Returns:**

        Complementary log-log, shape `(n,)`.
        """
        return jnp.log(-jnp.log1p(-mu))

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = 1 - \exp(-\exp(\eta))$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $1 - e^{-e^\eta}$, shape `(n,)`.
        """
        return -jnp.expm1(-jnp.exp(eta))

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = -1 / ((1 - \mu)\log(1 - \mu))$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        CLogLog link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = \exp(\eta - \exp(\eta))$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        CLogLog inverse-link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class LogLogLink(AbstractLink):
    r"""Log-log link $g(\mu) = -\log(-\log(\mu))$.

    The inverse link is $g^{-1}(\eta) = \exp(-\exp(-\eta))$.
    The derivative $g'(\mu)$ and inverse-link derivative $(g^{-1})'(\eta)$
    are computed via autodiff.

    This is the mirror of CLogLog: it is asymmetric in the opposite direction,
    approaching 0 faster than it approaches 1.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = -\log(-\log(\mu))$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries in $(0, 1)$.

        **Returns:**

        Log-log value, shape `(n,)`.
        """
        return -jnp.log(-jnp.log(mu))

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = \exp(-\exp(-\eta))$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $e^{-e^{-\eta}}$, shape `(n,)`.
        """
        return jnp.exp(-jnp.exp(-eta))

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = -1 / (\mu \log(\mu))$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        LogLog link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = \exp(-\eta - \exp(-\eta))$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        LogLog inverse-link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class SqrtLink(AbstractLink):
    r"""Square-root link $g(\mu) = \sqrt{\mu}$.

    The derivative is $g'(\mu) = 1 / (2\sqrt{\mu})$, the inverse link is
    $g^{-1}(\eta) = \eta^2$, and the inverse-link derivative is
    $(g^{-1})'(\eta) = 2\eta$.

    This is a variance-stabilising link for the Poisson family: it
    approximately stabilises the variance for count data.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \sqrt{\mu}$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $\geq 0$.

        **Returns:**

        $\sqrt{\mu}$, shape `(n,)`.
        """
        return jnp.sqrt(mu)

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = \eta^2$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`, entries $\geq 0$.

        **Returns:**

        $\eta^2$, shape `(n,)`.
        """
        return jnp.square(eta)

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = 1 / (2\sqrt{\mu})$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        $1/(2\sqrt{\mu})$, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = 2\eta$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $2\eta$, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class CauchitLink(AbstractLink):
    r"""Cauchit link $g(\mu) = \tan(\pi(\mu - 1/2))$.

    The inverse link is $g^{-1}(\eta) = 1/2 + \arctan(\eta)/\pi$.
    The derivative $g'(\mu)$ and inverse-link derivative $(g^{-1})'(\eta)$
    are computed via autodiff.

    This is a heavy-tailed alternative to the probit link for the Binomial
    family. It is robust to extreme observations near 0 or 1 because its tails
    decay as $1/\eta^2$ rather than exponentially.
    """

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \tan(\pi(\mu - 1/2))$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries in $(0, 1)$.

        **Returns:**

        Cauchit value, shape `(n,)`.
        """
        return jnp.tan(jnp.pi * (mu - 0.5))

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = 1/2 + \arctan(\eta) / \pi$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        $1/2 + \arctan(\eta)/\pi$, shape `(n,)`.
        """
        return 0.5 + jnp.arctan(eta) / jnp.pi

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = \pi / \cos^2(\pi(\mu - 1/2))$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        Cauchit link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta) = 1 / (\pi(1 + \eta^2))$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Cauchit inverse-link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)


class NBLink(AbstractLink):
    r"""Negative-binomial link $g(\mu) = \log(\alpha \mu / (1 + \alpha \mu))$.

    Here $\mu$ is the mean response and $\alpha > 0$ is the overdispersion
    parameter. The derivative is
    $g'(\mu) = 1 / (\mu (1 + \alpha \mu))$, the inverse link is
    $g^{-1}(\eta) = 1 / (\alpha \operatorname{expm1}(-\eta))$, and the
    inverse-link derivative is $(g^{-1})'(\eta)$ computed from that inverse
    expression.
    """

    alpha: Scalar

    def __init__(self, alpha: float = 1.0) -> None:
        r"""Construct a Negative Binomial link.

        **Arguments:**

        - `alpha`: overdispersion parameter $\alpha > 0$. Default `1.0`.
        """
        self.alpha = jnp.asarray(alpha)

    def __call__(self, mu: Array) -> Array:
        r"""Compute $g(\mu) = \log(\alpha \mu / (1 + \alpha \mu))$.

        The symbol $\alpha$ denotes the configured overdispersion parameter.

        **Arguments:**

        - `mu`: mean, shape `(n,)`, entries $> 0$.

        **Returns:**

        NB log-link value, shape `(n,)`.
        """
        # log(x) - log1p(x) is numerically more stable than log(x/(x+1))
        # at large mu*alpha, where x/(x+1) rounds to 1 and log returns 0.
        mu_alpha = mu * self.alpha
        return jnp.log(mu_alpha) - jnp.log1p(mu_alpha)

    def inverse(self, eta: Array) -> Array:
        r"""Compute $g^{-1}(\eta) = 1 / (\alpha \operatorname{expm1}(-\eta))$.

        The symbol $\alpha$ denotes the configured overdispersion parameter.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`, entries $< 0$.

        **Returns:**

        Mean $\mu$, shape `(n,)`.
        """
        return 1.0 / (self.alpha * jnp.expm1(-eta))

    def deriv(self, mu: Array) -> Array:
        r"""Compute $g'(\mu) = 1 / (\mu(1 + \alpha\mu))$ via autodiff.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.

        **Returns:**

        NB link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self, mu)

    def inverse_deriv(self, eta: Array) -> Array:
        r"""Compute $(g^{-1})'(\eta)$ via autodiff.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Inverse-link derivative, shape `(n,)`.
        """
        return _grad_per_sample(self.inverse, eta)
