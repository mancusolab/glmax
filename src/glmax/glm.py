# pattern: Functional Core

"""GLM specification noun and grammar verb factory."""

from __future__ import annotations

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

    def __init__(self, family: ExponentialDispersionFamily = Gaussian()) -> None:
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

        Tuple `(mu, variance, weight)` each of shape `(n,)`, where
        `weight` is the per-sample GLM working weight $w_i = 1 / (V(\mu_i) [g'(\mu_i)]^2)$.
        """
        mu, variance, weight = self.family.calc_weight(eta, disp, aux=aux)
        return mu, variance, weight

    def link_deriv(self, mu: ArrayLike) -> Array:
        r"""Evaluate the link derivative $g'(\mu)$.

        **Arguments:**

        - `mu`: mean response, shape `(n,)`.

        **Returns:**

        $g'(\mu)$, shape `(n,)`.
        """
        return self.family.glink.deriv(mu)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
        step_size: ScalarLike,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Apply one dispersion update step inside the IRLS loop.

        **Arguments:**

        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: observed response, shape `(n,)`.
        - `eta`: current linear predictor, shape `(n,)`.
        - `disp`: current EDM dispersion scalar.
        - `step_size`: IRLS step-size multiplier.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Updated dispersion scalar.
        """
        return self.family.update_dispersion(X, y, eta, disp, step_size, aux=aux)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
        aux: ScalarLike | None = None,
    ) -> Array:
        r"""Post-convergence dispersion estimate.

        Called once after IRLS convergence to produce the final $\hat\phi$.

        **Arguments:**

        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: observed response, shape `(n,)`.
        - `eta`: converged linear predictor, shape `(n,)`.
        - `disp`: IRLS EDM dispersion at convergence.
        - `aux`: optional family-specific auxiliary scalar.

        **Returns:**

        Final dispersion estimate scalar.
        """
        return self.family.estimate_dispersion(X, y, eta, disp, aux=aux)

    def canonicalize_dispersion(self, disp: ScalarLike) -> Array:
        r"""Map a raw dispersion value to its canonical stored form.

        Fixed-dispersion families (e.g. Poisson, Binomial) always return `1.0`.

        **Arguments:**

        - `disp`: raw dispersion value.

        **Returns:**

        Canonical dispersion scalar.
        """
        return self.family.canonical_dispersion(disp)

    def canonicalize_auxiliary(self, aux: ScalarLike | None) -> Array | None:
        r"""Map a raw auxiliary value to its canonical stored form.

        The `GLM` boundary always delegates to the active family's
        `canonical_auxiliary(...)` hook so the split `disp`/`aux` semantics stay
        centralized in the family contract.

        **Arguments:**

        - `aux`: optional raw family-specific auxiliary value.

        **Returns:**

        Canonical auxiliary scalar, or `None` when the active family does not
        use auxiliary state.

        """
        canonical_aux = self.family.canonical_auxiliary(aux)
        if canonical_aux is None:
            return None
        return jnp.asarray(canonical_aux)

    def canonicalize_params(self, disp: ScalarLike, aux: ScalarLike | None) -> tuple[Array, Array | None]:
        r"""Canonicalize `(disp, aux)` through the model boundary.

        This keeps warm-start normalization and family-compatibility checks on
        the `GLM` seam instead of duplicating them in fitting or inference
        kernels.

        **Arguments:**

        - `disp`: raw dispersion value.
        - `aux`: optional raw family-specific auxiliary value.

        **Returns:**

        Tuple `(canonical_disp, canonical_aux)` with family-aware canonical
        storage values.
        """
        return self.canonicalize_dispersion(disp), self.canonicalize_auxiliary(aux)

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        r"""Compute the scale helper used by inference estimators.

        This method remains an implementation helper for covariance scaling.
        It does not define the fitted `(disp, aux)` contract; fitted parameter
        semantics are carried by `canonicalize_params(...)` and stored `Params`.

        **Arguments:**

        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: observed response, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.

        **Returns:**

        Scalar $\hat\phi$ used to scale covariance matrices in inference.
        """
        return self.family.scale(X, y, mu)


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
