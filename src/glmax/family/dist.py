# pattern: Functional Core
import math

from abc import abstractmethod
from typing import ClassVar, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rdm
import jax.scipy.stats as jaxstats

from equinox import AbstractVar
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, ScalarLike

from .links import AbstractLink, IdentityLink, InverseLink, LogitLink, LogLink, NBLink, PowerLink


if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


def _validate_positive_finite_scalar(name: str, value: ScalarLike) -> Array:
    scalar = jnp.asarray(value)
    if scalar.ndim > 0 and scalar.size != 1:
        raise ValueError(f"{name} must be a scalar.")

    try:
        python_scalar = float(scalar)
    except TypeError:
        # Traced callers cannot branch on Python scalars; keep the value
        # numerically valid and let boundary callers handle deterministic errors.
        return jnp.clip(scalar, min=jnp.finfo(jnp.float64).tiny)

    if not math.isfinite(python_scalar) or python_scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {python_scalar}.")
    return scalar


def _nb_alpha_from_split(disp: ScalarLike, aux: ScalarLike | None) -> Array:
    alpha = aux if aux is not None else disp
    return _validate_positive_finite_scalar("NegativeBinomial alpha", alpha)


class ExponentialDispersionFamily(eqx.Module):
    r"""Abstract base for one-parameter exponential dispersion family distributions.

    A GLM models the conditional mean $\mu = \mathrm{E}(Y \mid X)$ via a link
    function $g$ such that $g(\mu) = \eta = X\beta$.  Subclasses implement the
    family-specific density, variance function, and split `(disp, aux)` handling.

    Concrete families: `Gaussian`, `Poisson`, `Binomial`, `NegativeBinomial`, `Gamma`.
    """

    glink: AbstractVar[AbstractLink]
    _links: AbstractClassVar[list[type[AbstractLink]]]
    _bounds: AbstractClassVar[tuple[float, float]]

    def __check_init__(self):
        if not any([isinstance(self.glink, link) for link in self._links]):
            raise ValueError(f"Link {self.glink} is invalid for Family {self}")

    @abstractmethod
    def negloglikelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Compute negative log-likelihood.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Scalar negative log-likelihood.
        """

    @abstractmethod
    def variance(self, mu: ArrayLike, disp: ScalarLike = 0.0, aux: ScalarLike | None = None) -> Array:
        r"""Variance function $V(\mu)$.

        **Arguments:**

        - `mu`: mean parameter, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Variance, shape `(n,)`.
        """

    @abstractmethod
    def sample(
        self,
        key: Array,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Draw samples from the family's distribution.

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
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array, Array]:
        r"""Compute IRLS weights.

        Clips $\mu$ to `_bounds` and variance to `(tiny, inf)` before
        computing $W = \operatorname{diag}(1 / (V(\mu) \cdot [g'(\mu)]^2))$.

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

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + y.mean()) / 2)

    def update_nuisance(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
        step_size: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array | None]:
        r"""Apply one nuisance-parameter update step inside the IRLS loop.

        Returns the updated `(disp, aux)` pair.  The base implementation is a
        no-op identity: it returns `(disp, aux)` unchanged.  Families that
        estimate a nuisance parameter override this method and update whichever
        slot they own — `disp` for EDM-dispersion families (e.g. Gaussian),
        `aux` for structural-parameter families (e.g. Negative Binomial).

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
        """Return the default ``(disp, aux)`` pair used to seed the IRLS loop.

        **Returns:**

        ``(jnp.asarray(1.0), None)`` for families without auxiliary state.
        """
        return jnp.asarray(1.0), None


class Gaussian(ExponentialDispersionFamily):
    r"""Gaussian (normal) exponential family.

    Models a continuous response $y \in \mathbb{R}$ with
    $y \mid \mu \sim \mathcal{N}(\mu, \sigma^2)$.

    The variance function is $V(\mu) = \sigma^2$ and the canonical link
    is the identity $g(\mu) = \mu$. Gaussian uses `disp` as EDM dispersion and ignores `aux`.
    """

    glink: AbstractLink = IdentityLink()
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, PowerLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def __init__(self, glink: AbstractLink = IdentityLink()) -> None:
        r"""**Arguments:**
        - `glink`: link function (default: `IdentityLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Gaussian negative log-likelihood.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: variance $\sigma^2$, scalar.  When `disp <= 0` (e.g. the
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

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0, aux: ScalarLike | None = None) -> Array:
        r"""Gaussian variance function $V(\mu) = \sigma^2$.

        When `disp <= 0` (e.g. the IRLS sentinel `0.0` on the first step
        before dispersion estimation is available), falls back to `1.0` so
        the first-step weight is unit weight and the WLS step is equivalent
        to OLS (uniform weights, correct for identity-link Gaussian).

        **Arguments:**

        - `mu`: mean, shape `(n,)`.
        - `disp`: variance $\sigma^2$, scalar.
        - `aux`: ignored.

        **Returns:**

        $\sigma^2 \cdot \mathbf{1}$, shape `(n,)`.
        """
        del aux
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return jnp.ones_like(mu) * safe_disp

    def update_nuisance(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
        step_size: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array | None]:
        r"""Compute RSS/df and return as the updated `(disp, aux)` pair.

        **Arguments:**

        - `X`: design matrix, shape `(n, p)`.
        - `y`: responses, shape `(n,)`.
        - `eta`: linear predictor at current iteration, shape `(n,)`.
        - `disp`: unused.
        - `step_size`: unused.
        - `aux`: ignored.

        **Returns:**

        `(hat_sigma_sq, None)` where $\hat{\sigma}^2 = \mathrm{RSS} / (n - p)$.
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
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Sample from $\mathcal{N}(\mu, \sigma^2)$ where $\mu = g^{-1}(\eta)$.

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


class Binomial(ExponentialDispersionFamily):
    r"""Binomial exponential family for binary responses.

    Models a binary response $y \in \{0, 1\}$ with
    $y \mid \mu \sim \mathrm{Bernoulli}(\mu)$, $\mu \in (0, 1)$.

    The variance function is $V(\mu) = \mu(1 - \mu)$ and the canonical link
    is the logit $g(\mu) = \log(\mu / (1 - \mu))$. Binomial fixes `disp = 1.0` and ignores `aux`.
    """

    glink: AbstractLink = LogitLink()
    _links: ClassVar[list[type[AbstractLink]]] = [
        LogitLink,
        LogLink,
        IdentityLink,
    ]  # Probit, Cauchy, LogC, CLogLog, LogLog
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, 1.0 - jnp.finfo(float).eps)

    def __init__(self, glink: AbstractLink = LogitLink()) -> None:
        r"""**Arguments:**
        - `glink`: link function (default: `LogitLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Binomial negative log-likelihood.

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

    def variance(self, mu: ArrayLike, disp: ScalarLike = 0.0, aux: ScalarLike | None = None) -> Array:
        del disp, aux
        return mu * (1 - mu)

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + 0.5) / 2.0)

    def sample(
        self,
        key: Array,
        eta: ArrayLike,
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


class Poisson(ExponentialDispersionFamily):
    r"""Poisson exponential family for count responses.

    Models a non-negative integer response $y \in \{0, 1, 2, \ldots\}$ with
    $y \mid \mu \sim \mathrm{Poisson}(\mu)$, $\mu > 0$.

    The variance function is $V(\mu) = \mu$ and the canonical link
    is the log $g(\mu) = \log(\mu)$. Poisson fixes `disp = 1.0` and ignores `aux`.
    """

    glink: AbstractLink = LogLink()
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink]  # Sqrt
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def __init__(self, glink: AbstractLink = LogLink()) -> None:
        r"""**Arguments:**
        - `glink`: link function (default: `LogLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Poisson negative log-likelihood.

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

    def variance(self, mu: ArrayLike, disp: ScalarLike = 0.0, aux: ScalarLike | None = None) -> Array:
        del disp, aux
        return mu

    def sample(
        self,
        key: Array,
        eta: ArrayLike,
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


class NegativeBinomial(ExponentialDispersionFamily):
    r"""Negative-binomial (NB-2) exponential family for overdispersed count data.

    Models a non-negative integer response via the NB-2 parameterisation
    $\mathrm{Var}(y \mid \mu) = \mu + \alpha \mu^2$, where $\alpha > 0$ is the
    overdispersion parameter. Uses a log link by default.
    Negative Binomial fixes `disp = 1.0` and uses `aux` as `alpha`.

    During the transition to the split contract, methods still accept legacy
    callers that pass `alpha` through `disp`; when `aux` is provided it always
    takes precedence, and whichever carrier is used is validated as a positive,
    finite `alpha`.
    """

    glink: AbstractLink = LogLink()
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, NBLink, PowerLink]  # CLogLog
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def __init__(self, glink: AbstractLink = LogLink()) -> None:
        r"""**Arguments:**
        - `glink`: link function (default: `LogLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Negative-binomial log-likelihood (numerically stable via `logaddexp`).

        Uses $r = 1/\alpha$ and the log-probability parameterization to avoid
        catastrophic cancellation for large counts.

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
        alpha = _nb_alpha_from_split(disp, aux)
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

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0, aux: ScalarLike | None = None) -> Array:
        alpha = _nb_alpha_from_split(disp, aux)
        return mu + alpha * (mu**2)

    def sample(
        self,
        key: Array,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Sample from $\mathrm{NB}(r, \mu)$ via Gamma-Poisson mixture.

        $\mathrm{NB}(r, \mu)$ is equivalent to $\mathrm{Poisson}(\lambda)$ where
        $\lambda \sim \mathrm{Gamma}(r, \mu/r)$.

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
        alpha = _nb_alpha_from_split(disp, aux)
        r = 1.0 / alpha
        # jax.random.gamma samples Gamma(a=r, scale=1); multiply by mu/r to get Gamma(a=r, scale=mu/r)
        gamma_sample = rdm.gamma(key1, r, shape=mu.shape) * (mu / r)
        return rdm.poisson(key2, lam=gamma_sample).astype(jnp.float64)

    def alpha_score_and_hessian(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ScalarLike
    ) -> tuple[Array, Array]:
        r"""Gradient and Hessian of the negative log-likelihood w.r.t. $\alpha$.

        **Arguments:**

        - `X`: design matrix (unused, kept for API symmetry).
        - `y`: observed counts, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `alpha`: overdispersion $\alpha > 0$, scalar.

        **Returns:**

        Tuple `(score, hessian)` — scalar gradient and scalar second derivative.
        """

        def _ll(alpha):
            return self.negloglikelihood(y, eta, alpha)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)
        return _alpha_score(alpha), _alpha_hess(alpha)  # .reshape((1,))

    def log_alpha_score_and_hessian(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, log_alpha: ScalarLike
    ) -> tuple[Array, Array]:
        r"""Gradient and Hessian of the negative log-likelihood w.r.t. $\log\alpha$.

        Differentiates in log-space to ensure $\alpha > 0$ throughout Newton steps.

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
            return self.negloglikelihood(y, eta, alpha_)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)

        return _alpha_score(log_alpha), _alpha_hess(log_alpha)  # .reshape((1,))

    def update_nuisance(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
        step_size: ScalarLike = 0.1,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array]:
        r"""Apply one Newton step on $\log\alpha$ and return `(1.0, new_alpha)`.

        Negative Binomial fixes `disp = 1.0` and updates `aux` (the
        overdispersion $\alpha$) each IRLS iteration via a Newton step in
        log-space.

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
        alpha = _nb_alpha_from_split(disp, aux)
        log_alpha = jnp.log(alpha)
        score, hess = self.log_alpha_score_and_hessian(X, y, eta, log_alpha)
        log_alpha_n = jnp.clip(
            log_alpha - step_size * (score / hess),
            min=jnp.log(jnp.asarray(1e-9)),
            max=jnp.log(jnp.asarray(1e9)),
        )
        return jnp.asarray(1.0), jnp.exp(log_alpha_n)

    def init_nuisance(self) -> tuple[Array, Array]:
        """Return the default ``(disp, aux)`` pair for NegativeBinomial.

        **Returns:**

        ``(jnp.asarray(1.0), jnp.asarray(0.1))`` — ``disp`` is always 1.0 for NB;
        ``aux`` is the initial overdispersion seed ``alpha = 0.1``.
        """
        return jnp.asarray(1.0), jnp.asarray(0.1)


class Gamma(ExponentialDispersionFamily):
    r"""Gamma exponential family with density

    $$f(y \mid \mu, \phi) = \frac{y^{1/\phi - 1} \exp(-y / (\mu\phi))}
    {\Gamma(1/\phi)(\mu\phi)^{1/\phi}}$$

    The mean is $\mu > 0$ and the variance is $\phi \mu^2$.

    The canonical link for Gamma is `InverseLink` ($g(\mu) = 1/\mu$).
    Gamma uses `disp` as EDM dispersion and ignores `aux`.
    """

    glink: AbstractLink = InverseLink()
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, InverseLink, LogLink]
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def __init__(self, glink: AbstractLink = InverseLink()) -> None:
        r"""**Arguments:**
        - `glink`: link function (default: `InverseLink()`).
        """
        self.glink = glink

    def negloglikelihood(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Gamma negative log-likelihood.

        Uses `jax.scipy.stats.gamma.logpdf` with shape $k = 1/\phi$ and
        scale $\theta = \mu \phi$.

        When `disp <= 0` (e.g. the IRLS sentinel `0.0` before dispersion
        estimation is wired in), falls back to `1.0` so the objective
        remains finite.

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

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0, aux: ScalarLike | None = None) -> Array:
        r"""Gamma variance $V(\mu) = \phi \mu^2$.

        When `disp <= 0` (e.g. the IRLS sentinel `0.0` on the first step
        before dispersion estimation is available), falls back to `1.0` so
        the first-step weights remain finite (equivalent to unit-variance
        Gamma weights, correct for the InverseLink canonical form).

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
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Sample from $\mathrm{Gamma}(k, \theta)$ where $k = 1/\phi$, $\theta = \mu\phi$.

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
