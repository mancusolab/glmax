# pattern: Functional Core

"""IRLS optimizer kernels and default fitter strategy."""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp

from jax import Array, lax

from ..data import GLMData
from .solve import AbstractLinearSolver, CholeskySolver
from .types import AbstractFitter, FitResult, Params


if TYPE_CHECKING:
    from jaxtyping import ArrayLike, ScalarLike

    from ..glm import GLM


__all__ = ["IRLSFitter"]


class _IRLSState(NamedTuple):
    beta: Array
    num_iters: int
    converged: Array
    disp: Array
    aux: Array
    objective: Array
    objective_delta: Array


def _irls(
    X: ArrayLike,
    y: ArrayLike,
    model: GLM,
    solver: AbstractLinearSolver,
    eta: ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
    offset_eta: ArrayLike = 0.0,
    disp_init: ScalarLike = 0.0,
    aux_init: ScalarLike = 0.0,
) -> _IRLSState:
    """IRLS to solve GLM."""
    _, p = X.shape

    def body_fun(val: tuple):
        likelihood_o, diff, num_iter, _beta_o, eta_o, disp_o, aux_o = val

        mu_k, g_deriv_k, weight = model.working_weights(eta_o, disp_o, aux_o)
        r = eta_o + g_deriv_k * (y - mu_k) * step_size - offset_eta

        beta = solver(X, r, weight)
        eta_n = X @ beta + offset_eta

        disp_n, aux_n = model.update_nuisance(X, y, eta_n, disp_o, step_size, aux_o)
        likelihood_n = -model.log_prob(y, eta_n, disp_n, aux_n)
        diff = likelihood_n - likelihood_o

        return likelihood_n, diff, num_iter + 1, beta, eta_n, disp_n, aux_n

    def cond_fun(val: tuple):
        likelihood_o, diff, num_iter, beta, eta, disp, aux = val
        del likelihood_o, beta, eta, disp, aux
        return jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)

    init_beta = jnp.zeros((p,))
    init_eta = eta + offset_eta
    init_likelihood = -model.log_prob(y, init_eta, disp_init, aux_init)
    init_tuple = (init_likelihood, jnp.inf, 0, init_beta, init_eta, disp_init, aux_init)

    objective, objective_delta, num_iters, beta, eta, disp, aux = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(objective_delta) < tol, num_iters <= max_iter)

    return _IRLSState(beta, num_iters, converged, disp, aux, objective, objective_delta)


class IRLSFitter(AbstractFitter, strict=True):
    r"""Iteratively Reweighted Least Squares (IRLS) fit strategy.

    The default `AbstractFitter` used by `glmax.fit(...)`. Runs a
    `lax.while_loop`-based IRLS algorithm and returns a `FitResult`.
    """

    solver: AbstractLinearSolver

    def __init__(self, solver: AbstractLinearSolver = CholeskySolver()) -> None:
        r"""**Arguments:**
        - `solver`: `AbstractLinearSolver` for each IRLS weighted least-squares
          step (default: `CholeskySolver()`).
        """
        self.solver = solver

    def __call__(
        self,
        model: GLM,
        data: GLMData,
        init: Params | None = None,
        max_iter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 1.0,
    ) -> FitResult:
        r"""Run IRLS to convergence and return a `FitResult`.

        **Arguments:**

        - `model`: `GLM` specification noun.
        - `data`: `GLMData` noun (weights not yet supported).
        - `init`: optional `Params` for warm-starting; `None` uses the family default.
        - `max_iter`: maximum number of IRLS iterations (default `1000`).
        - `tol`: convergence tolerance on the objective change (default `1e-3`).
        - `step_size`: IRLS update step-size multiplier (default `1.0`).

        **Returns:**

        `FitResult` with converged parameters, fit artifacts, and convergence metadata.

        **Raises:**

        - `TypeError`: if `data` is not a `GLMData` instance.
        - `ValueError`: if `data.weights` is set (not yet supported).
        """
        if not isinstance(data, GLMData):
            raise TypeError("fit(...) expects `data` to be a GLMData instance.")
        if data.weights is not None:
            raise ValueError("GLMData.weights is not supported yet.")
        X, y, offset, _ = data.canonical_arrays()

        default_disp, default_aux = model.init_nuisance()
        if init is not None:
            disp_init = jnp.asarray(init.disp)
            aux_init = jnp.asarray(init.aux) if init.aux is not None else default_aux
            init_eta = X @ jnp.asarray(init.beta) + offset
        else:
            disp_init = default_disp
            aux_init = default_aux
            init_eta = model.init_eta(y)

        state = _irls(
            X,
            y,
            model,
            self.solver,
            init_eta,
            max_iter,
            tol,
            step_size,
            offset,
            disp_init=disp_init,
            aux_init=aux_init,
        )
        beta, n_iter, converged, disp, aux, objective, objective_delta = state

        eta = X @ beta + offset
        mu, link_deriv, weight = model.working_weights(eta, disp, aux)
        score_residual = (y - mu) * link_deriv
        beta = jnp.ravel(beta)

        return FitResult(
            params=Params(beta=beta, disp=disp, aux=aux),
            X=X,
            y=y,
            eta=eta,
            mu=mu,
            glm_wt=weight,
            converged=converged,
            num_iters=n_iter,
            objective=objective,
            objective_delta=objective_delta,
            score_residual=score_residual,
        )
