# pattern: Functional Core

"""GLM specification noun and grammar verb factory."""

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jaxtyping import ArrayLike, ScalarLike

from .family import ExponentialDispersionFamily, Gaussian


class GLM(eqx.Module):
    r"""Generalized Linear Model specification noun.

    A [`glmax.GLM`][] holds the family that characterizes a model. It carries
    no fitted state and is constructed once via [`glmax.specify`][]. The
    linear solver lives on the [`glmax.AbstractFitter`][] strategy, not on the
    model itself.

    The model boundary exposes the split `(disp, aux)` contract consistently:
    $\phi$ is the exponential-dispersion-model dispersion scalar stored in
    `disp`, while `aux` carries optional family-specific state such as the
    Negative Binomial overdispersion parameter $\alpha$.
    """

    family: ExponentialDispersionFamily

    def __init__(self, family: ExponentialDispersionFamily = Gaussian()):
        r"""Construct a [`glmax.GLM`][] specification.

        **Arguments:**

        - `family`: exponential-dispersion family instance. Defaults to
          [`glmax.Gaussian`][].
        """
        self.family = family

    # ------------------------------------------------------------------
    # User-facing computations
    # ------------------------------------------------------------------

    def mean(self, eta: ArrayLike) -> Array:
        r"""Apply the inverse link to obtain fitted means.

        This computes $\mu = g^{-1}(\eta)$, where $\mu$ is the mean
        response, $\eta$ is the linear predictor, and $g$ is the link
        function associated with the active family.

        **Arguments:**

        - `eta`: linear predictor $\eta$, shape `(n,)`.

        **Returns:**

        Mean response vector $\mu = g^{-1}(\eta)$, shape `(n,)`.
        """
        return self.family.glink.inverse(eta)

    def log_prob(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Evaluate the total log-likelihood.

        This returns $\log p(y \mid \eta, \phi, a)$, where $y$ is the observed
        response, $\eta$ is the linear predictor, $\phi$ is the dispersion
        parameter, and $a$ is optional family-specific auxiliary state.

        **Arguments:**

        - `y`: observed response vector $y$, shape `(n,)`.
        - `eta`: linear predictor vector $\eta$, shape `(n,)`.
        - `disp`: dispersion scalar $\phi$.
        - `aux`: optional auxiliary scalar $a$ used by families with extra
          structural parameters.

        **Returns:**

        Scalar total log-likelihood.
        """
        return -self.family.negloglikelihood(y, eta, disp, aux=aux)

    def sample(
        self,
        key: Array,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Draw random samples from the fitted predictive distribution.

        Sampling is performed from the family distribution parameterized by
        $\eta$, $\phi$, and optional auxiliary state $a$, where
        $\mu = g^{-1}(\eta)$ is the implied mean response.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: linear predictor vector $\eta$, shape `(n,)`.
        - `disp`: dispersion scalar $\phi$.
        - `aux`: optional auxiliary scalar $a$.

        **Returns:**

        Sampled response values, shape `(n,)`.
        """
        return self.family.sample(key, eta, disp, aux=aux)

    # ------------------------------------------------------------------
    # Kernel interface (used by IRLS and inference internals)
    # ------------------------------------------------------------------

    def init_eta(self, y: ArrayLike) -> Array:
        r"""Compute the initial linear predictor for IRLS warm-start.

        The returned vector initializes the linear predictor $\eta$ from the
        observed response vector $y$.

        **Arguments:**

        - `y`: observed response vector $y$, shape `(n,)`.

        **Returns:**

        Initial $\eta$, shape `(n,)`.
        """
        return self.family.init_eta(y)

    def _working_terms(
        self,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array, Array]:
        mu = jnp.clip(self.mean(eta), *self.family._bounds)
        g_deriv = self.link_deriv(mu)
        variance = jnp.clip(jnp.asarray(self.family.variance(mu, disp, aux=aux)), min=jnp.finfo(float).tiny)
        weight = 1.0 / (variance * g_deriv**2)
        return mu, g_deriv, weight

    def working_weights(
        self,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array, Array]:
        r"""Compute IRLS working quantities at the current linear predictor.

        This returns $(\mu, g'(\mu), w)$, where $\mu = g^{-1}(\eta)$ is
        the mean response, $g'(\mu)$ is the link derivative evaluated at
        $\mu$, and
        $w = 1 / \left(\phi V(\mu) [g'(\mu)]^2\right)$ is the GLM working
        weight. Here $\phi$ is the dispersion scalar and $V(\mu)$ is the
        unit variance function of the family.

        **Arguments:**

        - `eta`: linear predictor vector $\eta$, shape `(n,)`.
        - `disp`: dispersion scalar $\phi$.
        - `aux`: optional auxiliary scalar $a$.

        **Returns:**

        Tuple `(mu, g_deriv, weight)` of arrays with shape `(n,)`.
        """
        return self.family.calc_weight(eta, disp, aux=aux)

    def link_deriv(self, mu: ArrayLike) -> Array:
        r"""Evaluate the link derivative $g'(\mu)$.

        The derivative is evaluated elementwise at the mean response $\mu$,
        where $g$ is the link function.

        **Arguments:**

        - `mu`: mean response vector $\mu$, shape `(n,)`.

        **Returns:**

        $g'(\mu)$, shape `(n,)`.
        """
        return self.family.glink.deriv(mu)

    def update_nuisance(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
        step_size: ScalarLike,
        aux: ScalarLike | None = None,
    ) -> tuple[Array, Array | None]:
        r"""Apply one nuisance-parameter update step inside the IRLS loop.

        Returns the updated `(disp, aux)` pair.  Families that estimate a
        nuisance parameter update whichever slot they own: `disp` for
        exponential-dispersion-model families with free $\phi$, and `aux` for
        structural-parameter families such as Negative Binomial with auxiliary
        parameter $\alpha$. Fixed-dispersion families return `(disp, aux)`
        unchanged.

        Here $X$ is the design matrix, $y$ is the response vector,
        $\eta$ is the current linear predictor, $\phi$ is the current
        dispersion scalar, and $a$ is optional auxiliary state.

        **Arguments:**

        - `X`: covariate matrix $X$, shape `(n, p)`.
        - `y`: observed response vector $y$, shape `(n,)`.
        - `eta`: current linear predictor vector $\eta$, shape `(n,)`.
        - `disp`: current dispersion scalar $\phi$.
        - `step_size`: IRLS step-size multiplier.
        - `aux`: optional auxiliary scalar $a$.

        **Returns:**

        Tuple `(new_disp, new_aux)`.
        """
        return self.family.update_nuisance(X, y, eta, disp, step_size, aux=aux)

    def init_nuisance(self) -> tuple[Array, Array | None]:
        r"""Return the default ``(disp, aux)`` pair used to seed the IRLS loop.

        Delegates to the active family so the caller does not need to distinguish
        between families at the kernel level.

        The returned pair is $(\phi_0, a_0)$, where $\phi_0$ is the
        initial dispersion value and $a_0$ is the initial auxiliary state.

        **Returns:**

        ``(default_disp, default_aux)`` where ``default_aux`` is ``None`` for
        families without auxiliary state.
        """
        return self.family.init_nuisance()


def specify(
    family: ExponentialDispersionFamily = Gaussian(),
) -> GLM:
    r"""Construct a GLM specification.

    This is the canonical factory for [`glmax.GLM`][]. The returned model
    holds only the family specification; it does not hold fitted
    coefficients or solver state.

    **Arguments:**

    - `family`: exponential-dispersion family instance. Defaults to
      [`glmax.Gaussian`][].

    **Returns:**

    A [`glmax.GLM`][] specification noun.
    """
    return GLM(family=family)
