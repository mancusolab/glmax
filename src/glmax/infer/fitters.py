# pattern: Functional Core

from abc import abstractmethod
from typing import NamedTuple, Tuple

import equinox as eqx

from jax import Array, lax, numpy as jnp
from jaxtyping import ArrayLike, ScalarLike

from ..family.dist import ExponentialFamily
from .solve import AbstractLinearSolver


class IRLSState(NamedTuple):
    beta: Array
    num_iters: int
    converged: Array
    alpha: Array


class AbstractGLMFitter(eqx.Module, strict=True):
    r"""Abstract optimization-strategy contract for GLM fitting.

    **Arguments:**

    - Implementations receive validated fit-boundary tensors and optimization
      controls (`max_iter`, `tol`, `step_size`).

    **Returns:**

    - [`IRLSState`][] containing coefficient updates, convergence metadata,
      and dispersion outputs for the calling fit pipeline.

    **Failure Modes:**

    - Implementations may raise backend numerical errors when solver updates
      cannot be computed for the provided operator.
    """

    @abstractmethod
    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        family: ExponentialFamily,
        solver: AbstractLinearSolver,
        eta: ArrayLike,
        max_iter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 1.0,
        offset_eta: ArrayLike = 0.0,
        alpha_init: ScalarLike = 0.0,
    ) -> IRLSState:
        pass


class IRLSFitter(AbstractGLMFitter):
    r"""Default IRLS-based fitter strategy.

    **Arguments:**

    - Uses weighted least-squares updates with family-specific working responses.

    **Returns:**

    - [`IRLSState`][] produced by iterative reweighted least squares updates.

    **Failure Modes:**

    - May fail when linear solves diverge or operators are ill-conditioned.
    """

    def __call__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        family: ExponentialFamily,
        solver: AbstractLinearSolver,
        eta: ArrayLike,
        max_iter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 1.0,
        offset_eta: ArrayLike = 0.0,
        alpha_init: ScalarLike = 0.0,
    ) -> IRLSState:
        r"""Run IRLS updates for one fit call.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `y`: Response vector with shape `(n,)`.
        - `family`: Exponential-family model specification.
        - `solver`: Linear solver strategy for weighted updates.
        - `eta`: Initial linear predictor.
        - `max_iter`: Maximum IRLS iterations.
        - `tol`: Convergence tolerance on likelihood deltas.
        - `step_size`: Step size for iterative updates.
        - `offset_eta`: Optional linear-predictor offset.
        - `alpha_init`: Initial dispersion parameter.

        **Returns:**

        - [`IRLSState`][] with final coefficients, convergence metadata, and dispersion.

        **Failure Modes:**

        - May raise backend linear-algebra errors from solver calls.
        """
        n, p = X.shape

        def body_fun(val: Tuple):
            likelihood_o, diff, num_iter, beta_o, eta_o, alpha_o = val

            mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta_o, alpha_o)
            r = eta_o + g_deriv_k * (y - mu_k) * step_size - offset_eta

            beta = solver(X, r, weight)

            eta_n = X @ beta + offset_eta

            alpha_n = family.update_dispersion(X, y, eta_n, alpha_o, step_size)
            likelihood_n = family.negloglikelihood(X, y, eta_n, alpha_n)
            diff = likelihood_n - likelihood_o

            return likelihood_n, diff, num_iter + 1, beta, eta_n, alpha_n

        def cond_fun(val: Tuple):
            likelihood_o, diff, num_iter, beta, eta, alpha = val
            cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
            return cond_l

        init_beta = jnp.zeros((p,))
        init_tuple = (10000.0, 10000.0, 0, init_beta, eta + offset_eta, alpha_init)

        likelihood_n, diff, num_iters, beta, eta, alpha = lax.while_loop(cond_fun, body_fun, init_tuple)
        converged = jnp.logical_and(jnp.fabs(diff) < tol, num_iters <= max_iter)

        return IRLSState(beta, num_iters, converged, alpha)


def irls(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    solver: AbstractLinearSolver,
    eta: ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
    offset_eta: ArrayLike = 0.0,
    alpha_init: ScalarLike = 0.0,
) -> IRLSState:
    r"""Functional compatibility wrapper for IRLS fitter execution.

    **Arguments:**

    - Matches historical `infer.optimize.irls` parameters for compatibility.

    **Returns:**

    - [`IRLSState`][] from the default [`IRLSFitter`][] strategy.

    **Failure Modes:**

    - Follows the same numerical failure behavior as [`IRLSFitter`][].
    """
    return IRLSFitter()(X, y, family, solver, eta, max_iter, tol, step_size, offset_eta, alpha_init)
