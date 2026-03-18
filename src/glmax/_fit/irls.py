# pattern: Functional Core

"""IRLS optimizer kernels and default fitter strategy."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax.numpy as jnp

from jax import Array, lax
from jaxtyping import ArrayLike, ScalarLike

from ..data import GLMData
from ..family import NegativeBinomial
from ..glm import GLM
from .solve import AbstractLinearSolver, CholeskySolver
from .types import _canonicalize_init, AbstractFitter, FitResult, Params


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
    has_aux: bool = False,
    updates_aux: bool = False,
) -> _IRLSState:
    """IRLS to solve GLM."""
    _, p = X.shape

    def body_fun(val: Tuple):
        likelihood_o, diff, num_iter, _beta_o, eta_o, disp_o, aux_o = val
        aux_arg = aux_o if has_aux else None

        mu_k, g_deriv_k, weight = model.working_weights(eta_o, disp_o, aux_arg)
        r = eta_o + g_deriv_k * (y - mu_k) * step_size - offset_eta

        beta = solver(X, r, weight)
        eta_n = X @ beta + offset_eta

        if updates_aux:
            disp_n = disp_o
            aux_n = model.update_dispersion(X, y, eta_n, disp_o, step_size, aux_arg)
        else:
            disp_n = model.update_dispersion(X, y, eta_n, disp_o, step_size, aux_arg)
            aux_n = aux_o
        likelihood_n = -model.log_prob(y, eta_n, disp_n, aux_n if has_aux else None)
        diff = likelihood_n - likelihood_o

        return likelihood_n, diff, num_iter + 1, beta, eta_n, disp_n, aux_n

    def cond_fun(val: Tuple):
        likelihood_o, diff, num_iter, beta, eta, disp, aux = val
        del likelihood_o, beta, eta, disp, aux
        return jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)

    init_beta = jnp.zeros((p,))
    init_eta = eta + offset_eta
    init_likelihood = -model.log_prob(y, init_eta, disp_init, aux_init if has_aux else None)
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

        init_beta, init_disp, init_aux = _canonicalize_init(init, X.shape[1])
        raw_init_disp = init_disp if init_disp is not None else 1.0
        canonical_init_disp, canonical_init_aux = model.canonicalize_params(raw_init_disp, init_aux)
        init_eta = X @ init_beta + 0.0 if init_beta is not None else model.init_eta(y)
        has_aux = canonical_init_aux is not None
        aux_init = canonical_init_aux if canonical_init_aux is not None else jnp.asarray(0.0)
        updates_aux = isinstance(model.family, NegativeBinomial)

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
            disp_init=canonical_init_disp,
            aux_init=aux_init,
            has_aux=has_aux,
            updates_aux=updates_aux,
        )
        beta, n_iter, converged, irls_disp, irls_aux, objective, objective_delta = state

        eta = X @ beta + offset
        aux_arg = irls_aux if has_aux else None
        if updates_aux:
            disp = irls_disp
            aux = model.estimate_dispersion(X, y, eta, irls_disp, aux=aux_arg)
        else:
            disp = model.estimate_dispersion(X, y, eta, irls_disp, aux=aux_arg)
            aux = aux_arg
        mu = model.mean(eta)
        score_residual = (y - mu) * model.link_deriv(mu)
        beta = jnp.ravel(beta)
        canonical_disp, canonical_aux = model.canonicalize_params(disp, aux)
        _, _, weight = model.working_weights(eta, canonical_disp, canonical_aux)

        return FitResult(
            params=Params(beta=beta, disp=canonical_disp, aux=canonical_aux),
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
