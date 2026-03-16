# pattern: Functional Core

"""GLM specification noun and grammar verb factory."""

from __future__ import annotations

import equinox as eqx

from jax import Array
from jaxtyping import ArrayLike, ScalarLike

from .family.dist import ExponentialFamily, Gaussian


class GLM(eqx.Module):
    r"""Generalized Linear Model specification noun.

    A `GLM` holds the family that characterises a model. It carries no state
    (no fitted parameters, no data) and is constructed once via `specify`.
    The linear solver is part of the `AbstractFitter` strategy, not the model.

    **Arguments:**

    - `family`: `ExponentialFamily` instance (default: `Gaussian()`).
    """

    family: ExponentialFamily = Gaussian()

    # ------------------------------------------------------------------
    # User-facing computations
    # ------------------------------------------------------------------

    def mean(self, eta: ArrayLike) -> Array:
        r"""Inverse link: $\mu = g^{-1}(\eta)$."""
        return self.family.glink.inverse(eta)

    def log_prob(self, y: ArrayLike, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Log-likelihood $\log p(y \mid \eta, \phi)$."""
        return -self.family.negloglikelihood(y, eta, disp)

    def sample(self, key: Array, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        """Draw samples from the fitted distribution."""
        return self.family.sample(key, eta, disp)

    # ------------------------------------------------------------------
    # Kernel interface (used by IRLS and inference internals)
    # ------------------------------------------------------------------

    def init_eta(self, y: ArrayLike) -> Array:
        """Initial linear predictor for IRLS warm-start."""
        return self.family.init_eta(y)

    def working_weights(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> tuple[Array, Array, Array]:
        """IRLS working weights: returns ``(mu, variance, weight)``."""
        return self.family.calc_weight(eta, disp)

    def link_deriv(self, mu: ArrayLike) -> Array:
        """Link derivative $g'(\mu)$."""
        return self.family.glink.deriv(mu)

    def update_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
        step_size: ScalarLike,
    ) -> Array:
        """Per-step dispersion update used inside the IRLS loop."""
        return self.family.update_dispersion(X, y, eta, disp, step_size)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
    ) -> Array:
        """Post-convergence dispersion estimate."""
        return self.family.estimate_dispersion(X, y, eta, disp)

    def canonicalize_dispersion(self, disp: ScalarLike) -> Array:
        """Map a raw dispersion value to its canonical stored form."""
        return self.family.canonical_dispersion(disp)

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        """Dispersion scale factor $\phi$ used by inference estimators."""
        return self.family.scale(X, y, mu)


def specify(
    family: ExponentialFamily = Gaussian(),
) -> GLM:
    r"""Construct a GLM specification.

    **Arguments:**

    - `family`: `ExponentialFamily` distribution (default: `Gaussian()`).

    **Returns:**

    A `GLM` specification noun.
    """
    return GLM(family=family)
