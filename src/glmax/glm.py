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
    """

    family: ExponentialFamily

    def __init__(self, family: ExponentialFamily = Gaussian()) -> None:
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

    def log_prob(self, y: ArrayLike, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Evaluate the log-likelihood $\log p(y \mid \eta, \phi)$.

        **Arguments:**

        - `y`: observed response, shape `(n,)`.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion scalar (default `0.0`; ignored by fixed-dispersion families).

        **Returns:**

        Per-sample log-likelihood, shape `(n,)`.
        """
        return -self.family.negloglikelihood(y, eta, disp)

    def sample(self, key: Array, eta: ArrayLike, disp: ScalarLike = 0.0) -> Array:
        r"""Draw random samples from the fitted predictive distribution.

        **Arguments:**

        - `key`: JAX PRNG key.
        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion scalar (default `0.0`; ignored by fixed-dispersion families).

        **Returns:**

        Sampled response values, shape `(n,)`.
        """
        return self.family.sample(key, eta, disp)

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

    def working_weights(self, eta: ArrayLike, disp: ScalarLike = 0.0) -> tuple[Array, Array, Array]:
        r"""Compute IRLS working quantities at the current linear predictor.

        **Arguments:**

        - `eta`: linear predictor, shape `(n,)`.
        - `disp`: dispersion scalar (default `0.0`).

        **Returns:**

        Tuple `(mu, variance, weight)` each of shape `(n,)`, where
        `weight` is the per-sample GLM working weight $w_i = 1 / (V(\mu_i) [g'(\mu_i)]^2)$.
        """
        return self.family.calc_weight(eta, disp)

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
    ) -> Array:
        r"""Apply one dispersion update step inside the IRLS loop.

        **Arguments:**

        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: observed response, shape `(n,)`.
        - `eta`: current linear predictor, shape `(n,)`.
        - `disp`: current dispersion scalar.
        - `step_size`: IRLS step-size multiplier.

        **Returns:**

        Updated dispersion scalar.
        """
        return self.family.update_dispersion(X, y, eta, disp, step_size)

    def estimate_dispersion(
        self,
        X: ArrayLike,
        y: ArrayLike,
        eta: ArrayLike,
        disp: ScalarLike,
    ) -> Array:
        r"""Post-convergence dispersion estimate.

        Called once after IRLS convergence to produce the final $\hat\phi$.

        **Arguments:**

        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: observed response, shape `(n,)`.
        - `eta`: converged linear predictor, shape `(n,)`.
        - `disp`: IRLS dispersion at convergence.

        **Returns:**

        Final dispersion estimate scalar.
        """
        return self.family.estimate_dispersion(X, y, eta, disp)

    def canonicalize_dispersion(self, disp: ScalarLike) -> Array:
        r"""Map a raw dispersion value to its canonical stored form.

        Fixed-dispersion families (e.g. Poisson, Binomial) always return `1.0`.

        **Arguments:**

        - `disp`: raw dispersion value.

        **Returns:**

        Canonical dispersion scalar.
        """
        return self.family.canonical_dispersion(disp)

    def scale(self, X: ArrayLike, y: ArrayLike, mu: ArrayLike) -> Array:
        r"""Compute the dispersion scale factor $\phi$ for inference estimators.

        **Arguments:**

        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: observed response, shape `(n,)`.
        - `mu`: fitted means, shape `(n,)`.

        **Returns:**

        Scalar $\hat\phi$ used to scale covariance matrices in inference.
        """
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
