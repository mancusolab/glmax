# pattern: Functional Core
from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rdm
import jax.scipy.stats as jaxstats

from jax import lax
from jax.scipy.special import gammaln
from jaxtyping import Array, ArrayLike, ScalarLike

from .links import AbstractLink, IdentityLink, InverseLink, LogitLink, LogLink, NBLink, PowerLink


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

    glink: AbstractLink
    _links: ClassVar[list[type[AbstractLink]]]
    _bounds: ClassVar[tuple[float, float]]

    def __check_init__(self):
        if not any([isinstance(self.glink, link) for link in self._links]):
            raise ValueError(f"Link {self.glink} is invalid for Family {self}")

    @abstractmethod
    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        """Compute the dispersion scale factor.

        **Arguments:**

        - `X`: design matrix, shape `(n, p)`.
        - `y`: observed responses, shape `(n,)`.
        - `mu`: mean parameter, shape `(n,)`.

        **Returns:**

        Scalar dispersion estimate.
        """

    @abstractmethod
    def negloglikelihood(self, y: ArrayLike, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Compute negative log-likelihood.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.

        **Returns:**

        Scalar negative log-likelihood.
        """

    @abstractmethod
    def variance(self, mu: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Variance function $V(\mu)$.

        **Arguments:**

        - `mu`: mean parameter, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.

        **Returns:**

        Variance, shape `(n,)`.
        """

    @abstractmethod
    def sample(self, key: Array, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Draw samples from the family's distribution.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.

        **Returns:**

        Samples, shape `(n,)`, same dtype as JAX float default.
        """

    def score(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        """
        For canonical link, this is X^t (y - mu)/phi, phi is the self.scale
        """
        return -X.T @ (y - mu) / self.scale(X, y, mu)

    def calc_weight(
        self,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
    ) -> tuple[Array, Array, Array]:
        r"""Compute IRLS weights.

        Clips $\mu$ to `_bounds` and variance to `(tiny, inf)` before
        computing $W = \operatorname{diag}(1 / (V(\mu) \cdot [g'(\mu)]^2))$.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion parameter, scalar.

        **Returns:**

        Three-tuple `(mu, variance, weight)`, each shape `(n,)`.
        """
        mu = jnp.clip(self.glink.inverse(eta), *self._bounds)
        v = jnp.clip(self.variance(mu, disp), min=jnp.finfo(float).tiny)
        w = 1.0 / (v * self.glink.deriv(mu) ** 2)
        return mu, v, w

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
        return self.canonical_dispersion(0.0)

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
        return self.canonical_dispersion(0.0)

    def canonical_dispersion(self, disp: ScalarLike = 0.0) -> Array:
        del disp
        return jnp.asarray(0.0)


class Gaussian(ExponentialFamily):
    """
    By explicitly write phi (here is sigma^2),
    we can treat normal distribution as one-parameter EF
    """

    glink: AbstractLink = IdentityLink()
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, PowerLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        resid = jnp.sum(jnp.square(mu - y))
        df = y.shape[0] - X.shape[1]
        phi = resid / df
        return phi

    def negloglikelihood(self, y: ArrayLike, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Gaussian negative log-likelihood.

        **Arguments:**

        - `y`: observed responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: variance $\sigma^2$, scalar.  When `disp <= 0` (e.g. the
          IRLS sentinel `0.0` before dispersion estimation is wired in),
          falls back to `1.0` so the objective remains finite and comparable.

        **Returns:**

        Scalar negative log-likelihood.
        """
        mu = self.glink.inverse(eta)
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return -jnp.sum(jaxstats.norm.logpdf(y, mu, jnp.sqrt(safe_disp)))

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Gaussian variance function $V(\mu) = \sigma^2$.

        When `disp <= 0` (e.g. the IRLS sentinel `0.0` on the first step
        before dispersion estimation is available), falls back to `1.0` so
        the first-step weight is unit weight and the WLS step is equivalent
        to OLS (uniform weights, correct for identity-link Gaussian).

        **Arguments:**

        - `mu`: mean, shape `(n,)`.
        - `disp`: variance $\sigma^2$, scalar.

        **Returns:**

        $\sigma^2 \cdot \mathbf{1}$, shape `(n,)`.
        """
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return jnp.ones_like(mu) * safe_disp

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.01,
        step_size: ScalarLike = 1.0,
    ) -> Array:
        r"""Compute RSS/df as the Gaussian dispersion estimate.

        **Arguments:**

        - `X`: design matrix, shape `(n, p)`.
        - `y`: responses, shape `(n,)`.
        - `eta`: linear predictor at current iteration, shape `(n,)`.
        - `alpha`: unused.
        - `step_size`: unused.

        **Returns:**

        $\hat{\sigma}^2 = \mathrm{RSS} / (n - p)$, scalar.  When $n \le p$
        (saturated or over-parameterised design), the denominator is clamped to
        1 to keep the result finite and non-negative.
        """
        mu = self.glink.inverse(eta)
        n, p = X.shape
        df = jnp.maximum(n - p, 1)
        return jnp.sum((y - mu) ** 2) / df

    def sample(self, key: Array, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Sample from $\mathcal{N}(\mu, \sigma^2)$ where $\mu = g^{-1}(\eta)$.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: variance $\sigma^2$, scalar.

        **Returns:**

        Gaussian samples, shape `(n,)`.
        """
        mu = self.glink.inverse(eta)
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return mu + rdm.normal(key, shape=mu.shape) * jnp.sqrt(safe_disp)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        **kwargs,
    ) -> Array:
        r"""Estimate $\sigma^2 = \mathrm{RSS} / (n - p)$.

        Delegates to `update_dispersion`.

        **Note on estimator:** $\mathrm{RSS} / (n - p)$ is the REML/unbiased
        estimator of $\sigma^2$ (not the MLE, which divides by $n$).  The
        denominator is clamped to at least 1 when $n \le p$ to keep the
        result finite.

        **Note on standard errors:** `FisherInfoError` phi-scaling is deferred
        to Phase 5.  Until Phase 5 lands, Gaussian standard errors returned by
        `infer()` are *not* phi-scaled (they assume $\phi = 1$).

        **Arguments:**

        - `X`: design matrix, shape `(n, p)`.
        - `y`: responses, shape `(n,)`.
        - `eta`: linear predictor at convergence, shape `(n,)`.
        - `disp`: unused (replaced by RSS/df).

        **Returns:**

        $\hat{\sigma}^2$, scalar.
        """
        del disp
        return self.update_dispersion(X, y, eta)

    def canonical_dispersion(self, disp: ScalarLike = 1.0) -> Array:
        r"""Return dispersion as-is (sigma^2 for Gaussian).

        **Arguments:**

        - `disp`: variance $\sigma^2$, scalar.

        **Returns:**

        `jnp.asarray(disp)`.
        """
        return jnp.asarray(disp)


class Binomial(ExponentialFamily):
    """
    default setting:
    glink = log(p/(1-p))
    glink_inv = 1/(1 + e^-x) # use log1p to calculate this
    glink_der = 1/(p*(1-p)) # use log trick to calculate this
    """

    glink: AbstractLink = LogitLink()
    _links: ClassVar[list[type[AbstractLink]]] = [
        LogitLink,
        LogLink,
        IdentityLink,
    ]  # Probit, Cauchy, LogC, CLogLog, LogLog
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, 1.0 - jnp.finfo(float).eps)

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(self, y: ArrayLike, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Binomial negative log-likelihood.

        **Arguments:**

        - `y`: binary responses in `{0, 1}`, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: unused (phi = 1 for Binomial).

        **Returns:**

        Scalar negative log-likelihood.
        """
        del disp
        return -jnp.sum(jaxstats.bernoulli.logpmf(y, self.glink.inverse(eta)))

    def variance(self, mu: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        return mu * (1 - mu)

    def init_eta(self, y: ArrayLike) -> Array:
        return self.glink((y + 0.5) / 2.0)

    def sample(self, key: Array, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Sample from $\mathrm{Bernoulli}(\mu)$ where $\mu = g^{-1}(\eta)$.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: unused (phi = 1 for Binomial).

        **Returns:**

        Binary samples, shape `(n,)`.
        """
        del disp
        mu = self.glink.inverse(eta)
        return rdm.bernoulli(key, p=mu, shape=mu.shape).astype(jnp.float64)

    def canonical_dispersion(self, disp: ScalarLike = 0.0) -> Array:
        r"""Canonical dispersion for Binomial is 1.0 (phi = 1).

        **Returns:**

        `jnp.asarray(1.0)`.
        """
        del disp
        return jnp.asarray(1.0)


class Poisson(ExponentialFamily):
    glink: AbstractLink = LogLink()
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink]  # Sqrt
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(self, y: ArrayLike, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Poisson negative log-likelihood.

        **Arguments:**

        - `y`: count responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: unused (phi = 1 for Poisson).

        **Returns:**

        Scalar negative log-likelihood.
        """
        del disp
        return -jnp.sum(jaxstats.poisson.logpmf(y, self.glink.inverse(eta)))

    def variance(self, mu: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        return mu

    def sample(self, key: Array, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Sample from $\mathrm{Poisson}(\mu)$ where $\mu = g^{-1}(\eta)$.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: unused (phi = 1 for Poisson).

        **Returns:**

        Count samples, shape `(n,)`.
        """
        del disp
        lam = self.glink.inverse(eta)
        return rdm.poisson(key, lam=lam).astype(jnp.float64)

    def canonical_dispersion(self, disp: ScalarLike = 0.0) -> Array:
        r"""Canonical dispersion for Poisson is 1.0 (phi = 1).

        **Returns:**

        `jnp.asarray(1.0)`.
        """
        del disp
        return jnp.asarray(1.0)


class NegativeBinomial(ExponentialFamily):
    """
    NB-2 method
    Notation: alpha = 1/r = 1.
    Now only use Log link (not the canonical link of NB)
    """

    glink: AbstractLink = LogLink()
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, LogLink, NBLink, PowerLink]  # CLogLog
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        return jnp.asarray(1.0)

    def negloglikelihood(self, y: ArrayLike, eta: ArrayLike, disp: ScalarLike = 0.1) -> Array:
        r"""Negative-binomial log-likelihood (numerically stable via `logaddexp`).

        Uses $r = 1/\alpha$ and the log-probability parameterization to avoid
        catastrophic cancellation for large counts.

        **Precondition:** `disp > 0`.  When `disp` is zero or below-`tiny`
        (e.g. passed as the IRLS sentinel `0.0` before dispersion estimation
        is wired in), it is silently clipped to `jnp.finfo(jnp.float64).tiny`
        so the result remains finite.  Callers that rely on meaningful
        likelihood values must supply a positive `disp`.

        **Arguments:**

        - `y`: count responses, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: overdispersion $\alpha > 0$, scalar.

        **Returns:**

        Scalar negative log-likelihood.
        """
        # Defensive guard added alongside Gamma work (Phase 4): clip prevents
        # log(0) when disp arrives as the IRLS sentinel 0.0 before estimation.
        disp = jnp.maximum(jnp.asarray(disp), jnp.finfo(jnp.float64).tiny)
        log_r = -jnp.log(disp)
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

    def variance(self, mu: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        return mu + disp * (mu**2)

    def sample(self, key: Array, eta: ArrayLike, disp: ScalarLike = 0.1) -> Array:
        r"""Sample from $\mathrm{NB}(r, \mu)$ via Gamma-Poisson mixture.

        $\mathrm{NB}(r, \mu)$ is equivalent to $\mathrm{Poisson}(\lambda)$ where
        $\lambda \sim \mathrm{Gamma}(r, \mu/r)$.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: overdispersion $\alpha > 0$, scalar.

        **Returns:**

        Count samples, shape `(n,)`.
        """
        # Defensive guard added alongside Gamma work (Phase 4): hard-fail at
        # boundary so sampling never silently draws from an ill-defined r = inf.
        if float(jnp.asarray(disp)) <= 0:
            raise ValueError(f"NegativeBinomial.sample: disp must be > 0, got {disp}")
        key1, key2 = rdm.split(key)
        mu = self.glink.inverse(eta)
        r = 1.0 / disp
        # jax.random.gamma samples Gamma(a=r, scale=1); multiply by mu/r to get Gamma(a=r, scale=mu/r)
        gamma_sample = rdm.gamma(key1, r, shape=mu.shape) * (mu / r)
        return rdm.poisson(key2, lam=gamma_sample).astype(jnp.float64)

    def alpha_score_and_hessian(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, alpha: ScalarLike
    ) -> tuple[Array, Array]:
        """
        internally take exponential such as to take derivative wrt 1/alpha
        """

        def _ll(alpha):
            return self.negloglikelihood(y, eta, alpha)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)
        return _alpha_score(alpha), _alpha_hess(alpha)  # .reshape((1,))

    def log_alpha_score_and_hessian(
        self, X: ArrayLike, y: ArrayLike, eta: ArrayLike, log_alpha: ScalarLike
    ) -> tuple[Array, Array]:
        """
        internally take exponential such as to take derivative wrt 1/alpha
        """

        def _ll(log_alpha_):
            alpha_ = jnp.exp(log_alpha_)
            return self.negloglikelihood(y, eta, alpha_)

        _alpha_score = jax.grad(_ll)
        _alpha_hess = jax.hessian(_ll)

        return _alpha_score(log_alpha), _alpha_hess(log_alpha)  # .reshape((1,))

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.1,
        step_size: ScalarLike = 0.1,
    ) -> Array:
        # TODO: update alpha such that it is lower bounded by 1e-6 (used to be 1e-8)
        #   should have either parameter or smarter update on Manifold
        log_alpha = jnp.log(alpha)
        score, hess = self.log_alpha_score_and_hessian(X, y, eta, log_alpha)
        log_alpha_n = jnp.clip(
            log_alpha - step_size * (score / hess),
            min=jnp.log(jnp.asarray(1e-9)),
            max=jnp.log(jnp.asarray(1e9)),
        )

        return jnp.exp(log_alpha_n)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        alpha: ScalarLike = 0.1,
        step_size=0.1,
        tol=1e-3,
        max_iter=1000,
    ) -> Array:
        alpha = jnp.asarray(alpha)

        def body_fun(val: tuple):
            diff, num_iter, alpha_o = val
            log_alpha_o = jnp.log(alpha_o)
            score, hess = self.log_alpha_score_and_hessian(X, y, eta, log_alpha_o)
            log_alpha_n = jnp.clip(
                log_alpha_o - step_size * (score / hess),
                min=jnp.log(jnp.asarray(1e-9)),
                max=jnp.log(jnp.asarray(1e9)),
            )
            diff = jnp.exp(log_alpha_n) - jnp.exp(log_alpha_o)

            return diff, num_iter + 1, jnp.exp(log_alpha_n)

        def cond_fun(val: tuple):
            diff, num_iter, alpha_o = val
            cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
            return cond_l

        init_tuple = (10000.0, 0, alpha)
        diff, num_iters, alpha = lax.while_loop(cond_fun, body_fun, init_tuple)

        return alpha

    def canonical_dispersion(self, disp: ScalarLike = 0.0) -> Array:
        return jnp.asarray(disp)


class Gamma(ExponentialFamily):
    r"""Gamma exponential family with density

    $$f(y \mid \mu, \phi) = \frac{y^{1/\phi - 1} \exp(-y / (\mu\phi))}
    {\Gamma(1/\phi)(\mu\phi)^{1/\phi}}$$

    The mean is $\mu > 0$ and the variance is $\phi \mu^2$.

    The canonical link for Gamma is `InverseLink` ($g(\mu) = 1/\mu$).
    `estimate_dispersion` is a no-op in this release; dispersion estimation is
    deferred to a future design.
    """

    glink: AbstractLink = InverseLink()
    _links: ClassVar[list[type[AbstractLink]]] = [IdentityLink, InverseLink, LogLink]
    _bounds: ClassVar[tuple[float, float]] = (jnp.finfo(float).tiny, jnp.inf)

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        r"""Return scale $\phi = 1$ (Gamma uses identity scale).

        **Arguments:**

        - `X`: design matrix, shape `(n, p)`.
        - `y`: responses, shape `(n,)`.
        - `mu`: mean, shape `(n,)`.

        **Returns:**

        Scalar `1.0`.
        """
        del X, y, mu
        return jnp.asarray(1.0)
        # Note: returning 1.0 means FisherInfoError will compute SE = 1.0 * inv(X'WX),
        # i.e. SE is not phi-scaled for Gamma. This is the correct deferred behaviour —
        # dispersion estimation for Gamma is a non-goal in this design. When a future
        # design adds Gamma dispersion estimation, this method will be updated.

    def negloglikelihood(self, y: ArrayLike, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
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

        **Returns:**

        Scalar negative log-likelihood.
        """
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        mu = jnp.clip(self.glink.inverse(eta), *self._bounds)
        k = 1.0 / safe_disp
        theta = mu * safe_disp
        return -jnp.sum(jaxstats.gamma.logpdf(y, a=k, scale=theta))

    def variance(self, mu: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Gamma variance $V(\mu) = \phi \mu^2$.

        When `disp <= 0` (e.g. the IRLS sentinel `0.0` on the first step
        before dispersion estimation is available), falls back to `1.0` so
        the first-step weights remain finite (equivalent to unit-variance
        Gamma weights, correct for the InverseLink canonical form).

        **Arguments:**

        - `mu`: mean, shape `(n,)`.
        - `disp`: dispersion $\phi$, scalar.

        **Returns:**

        $\phi \mu^2$, shape `(n,)`.
        """
        safe_disp = jnp.where(jnp.asarray(disp) > 0, disp, 1.0)
        return safe_disp * mu**2

    def canonical_dispersion(self, disp: ScalarLike = 1.0) -> Array:
        r"""Pass dispersion through unchanged.

        **Arguments:**

        - `disp`: dispersion $\phi$, scalar.

        **Returns:**

        `jnp.asarray(disp)`.
        """
        return jnp.asarray(disp)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        step_size: ScalarLike = 1.0,
    ) -> Array:
        r"""Return dispersion unchanged (estimation deferred).

        **Arguments:**

        - `X`: design matrix (unused).
        - `y`: responses (unused).
        - `eta`: linear predictor (unused).
        - `disp`: current dispersion estimate.
        - `step_size`: unused.

        **Returns:**

        `jnp.asarray(disp)` unchanged.
        """
        del X, y, eta, step_size
        return self.canonical_dispersion(disp)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 1.0,
        **kwargs,
    ) -> Array:
        r"""Return dispersion unchanged (estimation deferred).

        **Arguments:**

        - `X`: design matrix (unused).
        - `y`: responses (unused).
        - `eta`: linear predictor (unused).
        - `disp`: dispersion to return unchanged.

        **Returns:**

        `jnp.asarray(disp)`.
        """
        del X, y, eta
        return self.canonical_dispersion(disp)

    def sample(self, key: Array, eta: ArrayLike, disp: ScalarLike = 1.0) -> Array:
        r"""Sample from $\mathrm{Gamma}(k, \theta)$ where $k = 1/\phi$, $\theta = \mu\phi$.

        **Arguments:**

        - `key`: JAX PRNGKey.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion $\phi > 0$, scalar.

        **Returns:**

        Positive samples, shape `(n,)`.
        """
        mu = jnp.clip(self.glink.inverse(eta), *self._bounds)
        disp = jnp.clip(jnp.asarray(disp), min=jnp.finfo(float).tiny)
        k = 1.0 / disp
        theta = mu * disp
        return rdm.gamma(key, k, shape=mu.shape) * theta
