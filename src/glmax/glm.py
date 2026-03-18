# pattern: Functional Core

"""GLM specification noun and grammar verb factory."""

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jaxtyping import ArrayLike, ScalarLike

from .family import ExponentialDispersionFamily, Gaussian


class GLM(eqx.Module):
    r"""Generalized Linear Model specification noun.

    A `GLM` holds the family that characterises a model. It carries no state
    (no fitted parameters, no data) and is constructed once via `specify`.
    The linear solver is part of the `AbstractFitter` strategy, not the model.
    The model boundary exposes the split `(disp, aux)` contract consistently:
    `disp` is the EDM dispersion scalar, while `aux` carries optional
    family-specific state such as Negative Binomial `alpha`.
    """

    family: ExponentialDispersionFamily

    def __init__(self, family: ExponentialDispersionFamily = Gaussian()):
        r"""**Arguments:**
        - `family`: `ExponentialFamily` instance (default: `Gaussian()`).
        """
        self.family = family

    # ------------------------------------------------------------------
    # User-facing computations
    # ------------------------------------------------------------------

    def mean(self, eta: ArrayLike) -> Array:
        r"""Apply the inverse link to obtain fitted means.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.

        **Returns:**

        Mean response $\mu = g^{-1}(\eta)$, shape `(n,)`.
        """
        return self.family.glink.inverse(eta)

    def log_prob(
        self,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike = 0.0,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Evaluate the total log-likelihood $\log p(y \mid \eta, \mathrm{disp}, \mathrm{aux})$.

        **Arguments:**

        - `y`: observed response, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: EDM dispersion scalar.
        - `aux`: optional family-specific auxiliary scalar.

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

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: EDM dispersion scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Sampled response values, shape `(n,)`.
        """
        return self.family.sample(key, eta, disp, aux=aux)

    # ------------------------------------------------------------------
    # Kernel interface (used by IRLS and inference internals)
    # ------------------------------------------------------------------

    def init_eta(self, y: ArrayLike) -> Array:
        r"""Compute the initial linear predictor for IRLS warm-start.

        **Arguments:**

        - `y`: observed response, shape `(n,)`.

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

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: EDM dispersion scalar.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Tuple `(mu, g_deriv, weight)` each of shape `(n,)`, where
        `g_deriv` is the per-sample link derivative $g'(\mu_i)$ and
        `weight` is the per-sample GLM working weight $w_i = 1 / (V(\mu_i) [g'(\mu_i)]^2)$.
        """
        return self.family.calc_weight(eta, disp, aux=aux)

    def link_deriv(self, mu: ArrayLike) -> Array:
        r"""Evaluate the link derivative $g'(\mu)$.

        **Arguments:**

        - `mu`: mean response, shape `(n,)`.

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
        nuisance parameter update whichever slot they own — `disp` for
        EDM-dispersion families (e.g. Gaussian), `aux` for structural-parameter
        families (e.g. Negative Binomial).  Fixed-dispersion families
        (Poisson, Binomial) return `(disp, aux)` unchanged.

        **Arguments:**

        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: observed response, shape `(n,)`.
        - `eta`: current linear predictor, shape `(n,)`.
        - `disp`: current EDM dispersion scalar.
        - `step_size`: IRLS step-size multiplier.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Tuple `(new_disp, new_aux)`.
        """
        return self.family.update_nuisance(X, y, eta, disp, step_size, aux=aux)

    def init_nuisance(self) -> tuple[Array, Array | None]:
        r"""Return the default ``(disp, aux)`` pair used to seed the IRLS loop.

        Delegates to the active family so the caller does not need to distinguish
        between families at the kernel level.

        **Returns:**

        ``(default_disp, default_aux)`` where ``default_aux`` is ``None`` for
        families without auxiliary state.
        """
        return self.family.init_nuisance()


def specify(
    family: ExponentialDispersionFamily = Gaussian(),
) -> GLM:
    r"""Construct a GLM specification.

    **Arguments:**

    - `family`: `ExponentialFamily` distribution (default: `Gaussian()`).

    **Returns:**

    A `GLM` specification noun.
    """
    return GLM(family=family)
