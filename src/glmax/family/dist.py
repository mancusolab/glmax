# pattern: Functional Core

from abc import abstractmethod
from typing import ClassVar, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rdm
import jax.scipy.stats as jaxstats

from equinox import AbstractVar
from jax.scipy.special import betainc, gammaln, xlogy
from jaxtyping import Array, ScalarLike

from .links import (
    AbstractLink,
    CauchitLink,
    CLogLogLink,
    IdentityLink,
    InverseLink,
    LogitLink,
    LogLink,
    LogLogLink,
    NBLink,
    PowerLink,
    ProbitLink,
    SqrtLink,
)


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


class ExponentialDispersionFamily(eqx.Module):
    r"""Abstract base for one-parameter exponential dispersion family distributions.

    A GLM models the conditional mean $\mu = \mathrm{E}(Y \mid X)$ via a link
    function $g$ such that $g(\mu) = \eta = X \beta$.

    Subclasses implement the family-specific density, variance function, and split
    `(disp, aux)` handling, where `disp` stores the dispersion parameter $\phi$ and
    `aux` stores optional family-specific state $a$.
    """

    # instances of concrete classes should hold a link function
    glink: AbstractVar[AbstractLink]

    # helper class variable to discern if support is discrete or continuous
    is_discrete: AbstractClassVar[bool]

    # internal class variables to help with QC and invariant checking
    _links: AbstractClassVar[list[type[AbstractLink]]]
    _bounds: AbstractClassVar[tuple[float, float]]

    def __check_init__(self):
        if not any([isinstance(self.glink, link) for link in self._links]):
            raise ValueError(f"Link {self.glink} is invalid for Family {self}")

    @abstractmethod
    def negloglikelihood(
        self,
        y: Array,
        eta: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Compute negative log-likelihood.

        This returns the scalar objective
        $-\log p(y \mid \eta, \phi, a)$, where $y$ is the response vector,
        $\eta$ is the linear predictor, $\phi$ is the dispersion scalar,
        and $a$ is optional auxiliary state.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Scalar negative log-likelihood.
        """

    @abstractmethod
    def variance(self, mu: Array, disp: ScalarLike = 0.0, aux: ScalarLike | None = None) -> Array:
        r"""Variance function $V(\mu)$.

        This returns the family-specific variance expression evaluated at
        the mean response $\mu$. The dispersion scalar $\phi$ is passed via
        `disp` when the family uses it.

        **Arguments:**

        - `mu`: mean parameter, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Variance, shape `(n,)`.
        """

    @abstractmethod
    def cdf(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Cumulative distribution function for the family.

        This returns $F(y_i \mid \mu_i, \phi, a)$ elementwise, where $F$ is
        the fitted cumulative distribution function.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        CDF values $F(y_i \mid \mu_i)$, shape `(n,)`, values in `[0, 1]`.
        """

    @abstractmethod
    def deviance_contribs(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Per-observation deviance contributions for the family.

        The returned vector contains one deviance contribution $d_i$ for
        each observation $i$.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Non-negative deviance contributions, shape `(n,)`.
        """

    @abstractmethod
    def sample(
        self,
        key: Array,
        eta: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Draw samples from the family's distribution.

        Sampling uses the parameterization implied by the link-transformed
        mean $\mu = g^{-1}(\eta)$ together with the family's dispersion
        scalar $\phi$ and auxiliary state $a$ when present.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Samples, shape `(n,)`, same dtype as JAX float default.
        """

    def calc_weight(
        self,
        eta: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array, Array]:
        r"""Compute IRLS weights.

        Clips $\mu$ to `_bounds` and variance to `(tiny, inf)` before
        computing the working weights. This returns $(\mu, g'(\mu), w)$ where
        $w = 1 / \left(V(\mu) [g'(\mu)]^2\right)$. Here $\mu$ is the mean
        response, $V(\mu)$ is the family variance function, and $g$ is the
        link function.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Tuple `(mu, g_deriv, weight)` each of shape `(n,)`, where
        `g_deriv` is the per-sample link derivative $g'(\mu_i)$ and
        `weight` is the per-sample GLM working weight $w_i = 1 / (V(\mu_i) [g'(\mu_i)]^2)$.
        """
        mu = jnp.clip(self.glink.inverse(eta), *self._bounds)
        v = jnp.clip(jnp.asarray(self.variance(mu, disp, aux=aux)), min=jnp.finfo(float).tiny)
        g_deriv = self.glink.deriv(mu)
        w = 1.0 / (v * g_deriv**2)
        return mu, g_deriv, w

    def init_eta(self, y: Array) -> Array:
        return self.glink((y + y.mean()) / 2)

    def update_nuisance(
        self,
        X: Array,
        y: Array,
        eta: Array,
        disp: ScalarLike,
        step_size: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array | None]:
        r"""Apply one nuisance-parameter update step inside the IRLS loop.

        Returns the updated `(disp, aux)` pair.  The base implementation is a
        no-op identity: it returns `(disp, aux)` unchanged.  Families that
        estimate a nuisance parameter override this method and update whichever
        slot they own: `disp` for families with free dispersion $\phi$, and
        `aux` for structural-parameter families with auxiliary state $a$. The
        base implementation simply returns the incoming nuisance state
        unchanged.

        **Arguments:**

        - `X`: design matrix, shape `(n, p)`.
        - `y`: observed responses, shape `(n,)`.
        - `eta`: linear predictor at current iteration, shape `(n,)`.
        - `disp`: current EDM dispersion scalar.
        - `step_size`: IRLS step-size multiplier.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Tuple `(new_disp, new_aux)`.
        """
        del X, y, eta, step_size
        return jnp.asarray(disp), aux

    def init_nuisance(self) -> tuple[Array, Array | None]:
        r"""Return the default ``(disp, aux)`` pair used to seed the IRLS loop.

        This returns the initial nuisance state $(\phi_0, a_0)$.

        **Returns:**

        ``(jnp.asarray(1.0), None)`` for families without auxiliary state.
        """
        return jnp.asarray(1.0), None


class Gaussian(ExponentialDispersionFamily):
    r"""Gaussian (normal) exponential family.

    Models a continuous response $y \in \mathbb{R}$ with
    $y \mid \mu \sim \mathcal{N}(\mu, \sigma^2)$.

    The variance function is $V(\mu) = 1$ and the observation variance is
    $\phi V(\mu) = \phi$, where $\phi = \sigma^2$ is the Gaussian dispersion.
    The canonical link is the identity $g(\mu) = \mu$. Gaussian uses `disp`
    as the dispersion carrier and ignores `aux`.
    """

    glink: AbstractLink = IdentityLink()
    is_discrete: ClassVar[bool] = False
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, PowerLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def __init__(self, glink: AbstractLink = IdentityLink()) -> None:
        r"""Construct a Gaussian family.

        **Arguments:**

        - `glink`: link function (default: `IdentityLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: Array,
        eta: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Gaussian negative log-likelihood.

        This evaluates the Gaussian log density with mean
        $\mu = g^{-1}(\eta)$ and variance $\phi = \sigma^2$.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: variance $\phi = \sigma^2$, scalar. When `disp <= 0` (for
          example the
          IRLS sentinel `0.0` before dispersion estimation is wired in),
          falls back to `1.0` so the objective remains finite and comparable.
        - `aux`: ignored.

        **Returns:**

        Scalar negative log-likelihood.
        """
        del aux
        mu = self.glink.inverse(eta)
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return -jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(safe_disp)))

    def variance(self, mu: Array, disp: ScalarLike = 1.0, aux: ScalarLike | None = None) -> Array:
        r"""Gaussian variance term $\phi V(\mu) = \phi$.

        When `disp <= 0` (e.g. the IRLS sentinel `0.0` on the first step
        before dispersion estimation is available), falls back to `1.0` so
        the first-step weight is unit weight and the WLS step is equivalent
        to OLS (uniform weights, correct for identity-link Gaussian). For the
        Gaussian family the unit variance function is $V(\mu) = 1$, so the
        full variance is $\phi V(\mu) = \phi$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.
        - `disp`: dispersion scalar $\phi = \sigma^2$.
        - `aux`: ignored.

        **Returns:**

        $\sigma^2 \cdot \mathbf{1}$, shape `(n,)`.
        """
        del aux
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return jnp.ones_like(mu) * safe_disp

    def update_nuisance(
        self,
        X: Array,
        y: Array,
        eta: Array,
        disp: ScalarLike,
        step_size: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array | None]:
        r"""Compute RSS/df and return as the updated `(disp, aux)` pair.

        This sets $\hat{\phi} = \mathrm{RSS} / (n - p)$, where
        $\mathrm{RSS} = \sum_i (y_i - \mu_i)^2$.

        **Arguments:**

        - `X`: design matrix, shape `(n, p)`.
        - `y`: responses, shape `(n,)`.
        - `eta`: linear predictor at current iteration, shape `(n,)`.
        - `disp`: unused.
        - `step_size`: unused.
        - `aux`: ignored.

        **Returns:**

        `(\hat{\phi}, None)` where $\hat{\phi} = \mathrm{RSS} / (n - p)$.
        When $n \le p$ (saturated or over-parameterised design), the denominator
        is clamped to 1 to keep the result finite and non-negative.
        """
        del disp, step_size, aux
        mu = self.glink.inverse(eta)
        n, p = X.shape
        df = jnp.maximum(n - p, 1)
        return jnp.sum((y - mu) ** 2) / df, None

    def sample(
        self,
        key: Array,
        eta: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Sample from $\mathcal{N}(\mu, \sigma^2)$ where $\mu = g^{-1}(\eta)$.

        Here $\mu$ is the mean response and $\sigma^2 = \phi$ is the Gaussian
        variance.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: variance $\sigma^2$, scalar.
        - `aux`: ignored.

        **Returns:**

        Gaussian samples, shape `(n,)`.
        """
        del aux
        mu = self.glink.inverse(eta)
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return mu + rdm.normal(key, shape=mu.shape) * jnp.sqrt(safe_disp)

    def cdf(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Gaussian cumulative distribution function.

        This evaluates the Gaussian CDF with mean $\mu$ and variance
        $\phi = \sigma^2$.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.
        - `disp`: variance $\sigma^2$, scalar.
        - `aux`: ignored.

        **Returns:**

        CDF values, shape `(n,)`.
        """
        del aux
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return jaxstats.norm.cdf(y, loc=mu, scale=jnp.sqrt(safe_disp))

    def deviance_contribs(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Gaussian deviance contributions.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.
        - `disp`: ignored for unscaled deviance.
        - `aux`: ignored.

        **Returns:**

        Non-negative deviance contributions, shape `(n,)`.
        """
        del disp, aux
        return (y - mu) ** 2


class Binomial(ExponentialDispersionFamily):
    r"""Binomial exponential family for binary responses.

    Models a binary response $y \in \{0, 1\}$ with
    $y \mid \mu \sim \mathrm{Bernoulli}(\mu)$, $\mu \in (0, 1)$.

    The unit variance function is $V(\mu) = \mu(1 - \mu)$, so the model
    variance is $\phi V(\mu)$ with fixed $\phi = 1$. The canonical link is
    the logit $g(\mu) = \log(\mu / (1 - \mu))$. Binomial fixes `disp = 1.0`
    and ignores `aux`.
    """

    glink: AbstractLink = LogitLink()
    is_discrete: ClassVar[bool] = True
    _links: ClassVar[list[type[AbstractLink]]] = [
        LogitLink,
        ProbitLink,
        CauchitLink,
        CLogLogLink,
        LogLogLink,
        LogLink,
        IdentityLink,
    ]
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, 1.0 - jnp.finfo(float).eps)

    def __init__(self, glink: AbstractLink = LogitLink()) -> None:
        r"""Construct a Binomial family.

        **Arguments:**

        - `glink`: link function (default: `LogitLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: Array,
        eta: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Binomial negative log-likelihood.

        This uses the Bernoulli likelihood with success probability
        $\mu = g^{-1}(\eta)$.

        **Arguments:**

        - `y`: binary responses in `{0, 1}`, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: ignored; Binomial fixes `disp = 1.0`.
        - `aux`: ignored.

        **Returns:**

        Scalar negative log-likelihood.
        """
        del disp, aux
        return -jnp.sum(jaxstats.bernoulli.logpmf(y, self.glink.inverse(eta)))

    def variance(self, mu: Array, disp: ScalarLike = 0.0, aux: ScalarLike | None = None) -> Array:
        del disp, aux
        return mu * (1 - mu)

    def init_eta(self, y: Array) -> Array:
        return self.glink((y + 0.5) / 2.0)

    def sample(
        self,
        key: Array,
        eta: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Sample from $\mathrm{Bernoulli}(\mu)$ where $\mu = g^{-1}(\eta)$.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: ignored; Binomial fixes `disp = 1.0`.
        - `aux`: ignored.

        **Returns:**

        Binary samples, shape `(n,)`.
        """
        del disp, aux
        mu = self.glink.inverse(eta)
        return rdm.bernoulli(key, p=mu, shape=mu.shape).astype(jnp.float64)

    def cdf(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Bernoulli cumulative distribution function.

        **Arguments:**

        - `y`: binary responses in `{0, 1}`, shape `(n,)`.
        - `mu`: success probabilities, shape `(n,)`.
        - `disp`: ignored.
        - `aux`: ignored.

        **Returns:**

        CDF values, shape `(n,)`.
        """
        del disp, aux
        return jaxstats.bernoulli.cdf(y, p=mu)

    def deviance_contribs(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Binomial deviance contributions.

        **Arguments:**

        - `y`: binary responses in `{0, 1}`, shape `(n,)`.
        - `mu`: success probabilities, shape `(n,)`.
        - `disp`: ignored.
        - `aux`: ignored.

        **Returns:**

        Non-negative deviance contributions, shape `(n,)`.
        """
        del disp, aux
        y_ = y
        mu_ = jnp.clip(mu, *self._bounds)
        return 2.0 * (xlogy(y_, y_) - xlogy(y_, mu_) + xlogy(1.0 - y_, 1.0 - y_) - xlogy(1.0 - y_, 1.0 - mu_))


class Poisson(ExponentialDispersionFamily):
    r"""Poisson exponential family for count responses.

    Models a non-negative integer response $y \in \{0, 1, 2, \ldots\}$ with
    $y \mid \mu \sim \mathrm{Poisson}(\mu)$, $\mu > 0$.

    The unit variance function is $V(\mu) = \mu$, so the model variance is
    $\phi V(\mu)$ with fixed $\phi = 1$. The canonical link is
    $g(\mu) = \log(\mu)$. Poisson fixes `disp = 1.0` and ignores `aux`.
    """

    glink: AbstractLink = LogLink()
    is_discrete: ClassVar[bool] = True
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, SqrtLink]
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def __init__(self, glink: AbstractLink = LogLink()) -> None:
        r"""Construct a Poisson family.

        **Arguments:**

        - `glink`: link function (default: `LogLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: Array,
        eta: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Poisson negative log-likelihood.

        This uses the Poisson likelihood with rate $\mu = g^{-1}(\eta)$.

        **Arguments:**

        - `y`: count responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: ignored; Poisson fixes `disp = 1.0`.
        - `aux`: ignored.

        **Returns:**

        Scalar negative log-likelihood.
        """
        del disp, aux
        return -jnp.sum(jaxstats.poisson.logpmf(y, self.glink.inverse(eta)))

    def variance(self, mu: Array, disp: ScalarLike = 0.0, aux: ScalarLike | None = None) -> Array:
        del disp, aux
        return mu

    def sample(
        self,
        key: Array,
        eta: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Sample from $\mathrm{Poisson}(\mu)$ where $\mu = g^{-1}(\eta)$.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: ignored; Poisson fixes `disp = 1.0`.
        - `aux`: ignored.

        **Returns:**

        Count samples, shape `(n,)`.
        """
        del disp, aux
        lam = self.glink.inverse(eta)
        return rdm.poisson(key, lam=lam).astype(jnp.float64)

    def cdf(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Poisson cumulative distribution function.

        **Arguments:**

        - `y`: count responses, shape `(n,)`.
        - `mu`: fitted means (rates), shape `(n,)`.
        - `disp`: ignored.
        - `aux`: ignored.

        **Returns:**

        CDF values, shape `(n,)`.
        """
        del disp, aux
        return jaxstats.poisson.cdf(y, mu=mu)

    def deviance_contribs(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Poisson deviance contributions.

        **Arguments:**

        - `y`: count responses, shape `(n,)`.
        - `mu`: fitted means (rates), shape `(n,)`.
        - `disp`: ignored.
        - `aux`: ignored.

        **Returns:**

        Non-negative deviance contributions, shape `(n,)`.
        """
        del disp, aux
        y_ = y
        mu_ = jnp.clip(mu, *self._bounds)
        return 2.0 * (xlogy(y_, y_) - xlogy(y_, mu_) - (y_ - mu_))


class NegativeBinomial(ExponentialDispersionFamily):
    r"""Negative-binomial (NB-2) exponential family for overdispersed count data.

    Models a non-negative integer response via the NB-2 parameterization
    $\operatorname{Var}(Y \mid \mu) = \mu + \alpha \mu^2$, where $\alpha > 0$ is the
    overdispersion parameter. Uses a log link by default.
    Negative Binomial fixes `disp = 1.0` and uses `aux` as `alpha`.

    During the transition to the split contract, methods still accept legacy
    callers that pass `alpha` through `disp`; when `aux` is provided it always
    takes precedence, and whichever carrier is used is validated as a positive,
    finite `alpha`.
    """

    glink: AbstractLink = LogLink()
    is_discrete: ClassVar[bool] = True
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, NBLink, PowerLink]  # CLogLog
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def __init__(self, glink: AbstractLink = LogLink()) -> None:
        r"""Construct a Negative Binomial family.

        **Arguments:**

        - `glink`: link function (default: `LogLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: Array,
        eta: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Negative-binomial log-likelihood (numerically stable via `logaddexp`).

        Uses $r = 1/\alpha$ and the log-probability parameterization to avoid
        catastrophic cancellation for large counts. Here $\alpha$ is the
        overdispersion parameter and $r = 1 / \alpha$ is the corresponding
        count-shape parameter.

        **Precondition:** NB uses `aux` as `alpha > 0` and fixes `disp = 1.0`.
        When `aux` is omitted, the legacy `disp` carrier is still accepted
        during the phase transition and validated as the positive, finite
        `alpha` value.

        **Arguments:**

        - `y`: count responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: fixed at `1.0`; ignored when `aux` is provided.
        - `aux`: overdispersion $\alpha > 0$, scalar.

        **Returns:**

        Scalar negative log-likelihood.
        """
        alpha = aux
        log_r = -jnp.log(alpha)
        r = jnp.exp(log_r)
        # Compute log_mu in log-domain directly (glink is LogLink: inverse = exp(eta)).
        # Clipping eta avoids exp() overflow before log(), keeping log_mu finite.
        log_mu = jnp.clip(eta, jnp.log(self._bounds[0]), jnp.inf)
        log_mu_plus_r = jnp.logaddexp(log_mu, log_r)
        log_p = log_mu - log_mu_plus_r
        log1m_p = log_r - log_mu_plus_r
        term1 = gammaln(y + r) - gammaln(y + 1) - gammaln(r)
        term2 = r * log1m_p + y * log_p
        return -jnp.sum(term1 + term2)

    def variance(self, mu: Array, disp: ScalarLike = 1.0, aux: ScalarLike | None = None) -> Array:
        alpha = aux
        return mu + alpha * (mu**2)

    def sample(
        self,
        key: Array,
        eta: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Sample from $\mathrm{NB}(r, \mu)$ via Gamma-Poisson mixture.

        $\mathrm{NB}(r, \mu)$ is equivalent to $\mathrm{Poisson}(\lambda)$ where
        $\lambda \sim \mathrm{Gamma}(r, \mu/r)$. Here $r = 1 / \alpha$, where
        $\alpha$ is the overdispersion parameter.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: fixed at `1.0`; ignored when `aux` is provided.
        - `aux`: overdispersion $\alpha > 0$, scalar.

        **Returns:**

        Count samples, shape `(n,)`.
        """
        key1, key2 = rdm.split(key)
        mu = self.glink.inverse(eta)
        alpha = aux
        r = 1.0 / alpha
        # jax.random.gamma samples Gamma(a=r, scale=1); multiply by mu/r to get Gamma(a=r, scale=mu/r)
        gamma_sample = rdm.gamma(key1, r, shape=mu.shape) * (mu / r)
        return rdm.poisson(key2, lam=gamma_sample).astype(jnp.float64)

    def alpha_score_and_hessian(self, X: Array, y: Array, eta: Array, alpha: ScalarLike) -> tuple[Array, Array]:
        r"""Gradient and Hessian of the negative log-likelihood w.r.t. $\alpha$.

        Here $\alpha > 0$ is the Negative Binomial overdispersion parameter.

        **Arguments:**

        - `X`: design matrix (unused, kept for API symmetry).
        - `y`: observed counts, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `alpha`: overdispersion $\alpha > 0$, scalar.

        **Returns:**

        Tuple `(score, hessian)` — scalar gradient and scalar second derivative.
        """

        def _ll(alpha):
            return self.negloglikelihood(y, eta, aux=alpha)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)
        return _alpha_score(alpha), _alpha_hess(alpha)  # .reshape((1,))

    def log_alpha_score_and_hessian(self, X: Array, y: Array, eta: Array, log_alpha: ScalarLike) -> tuple[Array, Array]:
        r"""Gradient and Hessian of the negative log-likelihood w.r.t. $\log\alpha$.

        Differentiates in log-space to ensure $\alpha > 0$ throughout Newton
        steps. Here $\log \alpha$ is the unconstrained parameter used for
        Newton updates.

        **Arguments:**

        - `X`: design matrix (unused, kept for API symmetry).
        - `y`: observed counts, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `log_alpha`: $\log\alpha$, scalar.

        **Returns:**

        Tuple `(score, hessian)` — scalar gradient and scalar second derivative.
        """

        def _ll(log_alpha_):
            alpha_ = jnp.exp(log_alpha_)
            return self.negloglikelihood(y, eta, aux=alpha_)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)

        return _alpha_score(log_alpha), _alpha_hess(log_alpha)  # .reshape((1,))

    def update_nuisance(
        self,
        X: Array,
        y: Array,
        eta: Array,
        disp: ScalarLike,
        step_size: ScalarLike = 0.1,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array]:
        r"""Apply one Newton step on $\log\alpha$ and return `(1.0, new_alpha)`.

        Negative Binomial fixes `disp = 1.0` and updates `aux` (the
        overdispersion $\alpha$) each IRLS iteration via a Newton step in
        log-space. The returned pair is $(1.0, \alpha_{\text{new}})$ because
        Negative Binomial keeps `disp` fixed at `1.0` and stores
        overdispersion in `aux`.

        **Arguments:**

        - `X`: design matrix, shape `(n, p)`.
        - `y`: observed counts, shape `(n,)`.
        - `eta`: linear predictor at current iteration, shape `(n,)`.
        - `disp`: ignored; Negative Binomial fixes `disp = 1.0`.
        - `step_size`: Newton step damping factor (default `0.1`).
        - `aux`: current overdispersion $\alpha > 0$.

        **Returns:**

        `(1.0, new_alpha)`.
        """
        alpha = aux
        log_alpha = jnp.log(alpha)
        score, hess = self.log_alpha_score_and_hessian(X, y, eta, log_alpha)
        log_alpha_n = jnp.clip(
            log_alpha - step_size * (score / hess),
            min=jnp.log(jnp.asarray(1e-9)),
            max=jnp.log(jnp.asarray(1e9)),
        )
        return jnp.asarray(1.0), jnp.exp(log_alpha_n)

    def init_nuisance(self) -> tuple[Array, Array]:
        r"""Return the default ``(disp, aux)`` pair for NegativeBinomial.

        The default nuisance state is $(1.0, 0.1)$, where `aux = 0.1`
        initializes the overdispersion parameter $\alpha$.

        **Returns:**

        ``(jnp.asarray(1.0), jnp.asarray(0.1))`` — ``disp`` is always 1.0 for NB;
        ``aux`` is the initial overdispersion seed ``alpha = 0.1``.
        """
        return jnp.asarray(1.0), jnp.asarray(0.1)

    def cdf(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Negative-binomial cumulative distribution function.

        **Arguments:**

        - `y`: count responses, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.
        - `disp`: legacy alpha carrier (used if `aux` is `None`).
        - `aux`: overdispersion $\alpha > 0$, scalar.

        **Returns:**

        CDF values, shape `(n,)`.
        """
        alpha = aux
        r = 1.0 / alpha
        p_fail = r / (r + mu)
        return betainc(r, jnp.floor(y) + 1.0, p_fail)

    def deviance_contribs(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Negative-binomial deviance contributions.

        **Arguments:**

        - `y`: count responses, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.
        - `disp`: legacy alpha carrier (used if `aux` is `None`).
        - `aux`: overdispersion $\alpha > 0$, scalar.

        **Returns:**

        Non-negative deviance contributions, shape `(n,)`.
        """
        alpha = aux
        r = 1.0 / alpha
        y_ = y
        mu_ = jnp.clip(mu, *self._bounds)
        return 2.0 * (r * (jnp.log1p(mu_ / r) - jnp.log1p(y_ / r)) + xlogy(y_, y_) - xlogy(y_, mu_))


class Gamma(ExponentialDispersionFamily):
    r"""Gamma exponential family with density

    $$f(y \mid \mu, \phi) = \frac{y^{1/\phi - 1} \exp(-y / (\mu\phi))}
    {\Gamma(1/\phi)(\mu\phi)^{1/\phi}}$$

    The mean is $\mu > 0$ and the variance is $\phi V(\mu)$ with
    $V(\mu) = \mu^2$.

    The canonical link for Gamma is `InverseLink` ($g(\mu) = 1/\mu$).
    Gamma uses `disp` as EDM dispersion and ignores `aux`.
    """

    glink: AbstractLink = InverseLink()
    is_discrete: ClassVar[bool] = False
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, InverseLink, LogLink]
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def __init__(self, glink: AbstractLink = InverseLink()) -> None:
        r"""Construct a Gamma family.

        **Arguments:**

        - `glink`: link function (default: `InverseLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: Array,
        eta: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Gamma negative log-likelihood.

        Uses `jax.scipy.stats.gamma.logpdf` with shape $k = 1/\phi$ and
        scale $\theta = \mu \phi$.

        When `disp <= 0` (e.g. the IRLS sentinel `0.0` before dispersion
        estimation is wired in), falls back to `1.0` so the objective
        remains finite. The Gamma shape-scale parameterization uses
        $k = 1 / \phi$ and $\theta = \mu \phi$, where $\phi$ is the
        dispersion scalar.

        **Arguments:**

        - `y`: positive responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion $\phi > 0$, scalar.
        - `aux`: ignored.

        **Returns:**

        Scalar negative log-likelihood.
        """
        del aux
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        mu = jnp.clip(self.glink.inverse(eta), *self._bounds)
        k = 1.0 / safe_disp
        theta = mu * safe_disp
        return -jnp.sum(jaxstats.gamma.logpdf(y, a=k, scale=theta))

    def variance(self, mu: Array, disp: ScalarLike = 1.0, aux: ScalarLike | None = None) -> Array:
        r"""Gamma variance term $\phi V(\mu) = \phi \mu^2$.

        When `disp <= 0` (e.g. the IRLS sentinel `0.0` on the first step
        before dispersion estimation is available), falls back to `1.0` so
        the first-step weights remain finite. The unit variance function is
        $V(\mu) = \mu^2$, so the full variance is
        $\phi V(\mu) = \phi \mu^2$.

        **Arguments:**

        - `mu`: mean, shape `(n,)`.
        - `disp`: dispersion $\phi$, scalar.
        - `aux`: ignored.

        **Returns:**

        $\phi \mu^2$, shape `(n,)`.
        """
        del aux
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return safe_disp * mu**2

    def sample(
        self,
        key: Array,
        eta: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Sample from $\mathrm{Gamma}(k, \theta)$ where $k = 1/\phi$, $\theta = \mu\phi$.

        Here $k$ is the Gamma shape parameter and $\theta$ is the Gamma scale
        parameter.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion $\phi > 0$, scalar.
        - `aux`: ignored.

        **Returns:**

        Positive samples, shape `(n,)`.
        """
        del aux
        mu = jnp.clip(self.glink.inverse(eta), *self._bounds)
        disp = jnp.clip(jnp.asarray(disp), min=jnp.finfo(float).tiny)
        k = 1.0 / disp
        theta = mu * disp
        return rdm.gamma(key, k, shape=mu.shape) * theta

    def cdf(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Gamma cumulative distribution function.

        **Arguments:**

        - `y`: positive responses, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.
        - `disp`: dispersion $\phi > 0$, scalar.
        - `aux`: ignored.

        **Returns:**

        CDF values, shape `(n,)`.
        """
        del aux
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        mu_ = jnp.clip(mu, *self._bounds)
        k = 1.0 / safe_disp
        theta = mu_ * safe_disp
        return jaxstats.gamma.cdf(y, a=k, scale=theta)

    def deviance_contribs(
        self,
        y: Array,
        mu: Array,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Gamma deviance contributions.

        **Arguments:**

        - `y`: positive responses, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.
        - `disp`: ignored for unscaled deviance.
        - `aux`: ignored.

        **Returns:**

        Non-negative deviance contributions, shape `(n,)`.
        """
        del disp, aux
        y_ = y
        mu_ = mu
        r_ = (y_ - mu_) / mu_
        return 2.0 * (r_ - jnp.log1p(r_))
