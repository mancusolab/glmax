# pattern: Functional Core

"""IRLS optimizer kernels and default fitter strategy."""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax.numpy as jnp

from jax import Array, lax
from jaxtyping import ArrayLike, ScalarLike

from ..data import GLMData
from ..family.dist import ExponentialFamily
from ..glm import GLM
from .solve import AbstractLinearSolver
from .types import _canonicalize_init, FitResult, Fitter, Params


__all__ = ["irls", "IRLSFitter"]


class _IRLSState(NamedTuple):
    beta: Array
    num_iters: int
    converged: Array
    disp: Array
    objective: Array
    objective_delta: Array


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
    disp_init: ScalarLike = 0.0,
) -> _IRLSState:
    """IRLS to solve GLM."""
    n, p = X.shape

    def body_fun(val: Tuple):
        likelihood_o, diff, num_iter, beta_o, eta_o, disp_o = val

        mu_k, _v, weight = family.calc_weight(eta_o, disp_o)
        g_deriv_k = family.glink.deriv(mu_k)
        r = eta_o + g_deriv_k * (y - mu_k) * step_size - offset_eta

        beta = solver(X, r, weight)
        eta_n = X @ beta + offset_eta

        disp_n = family.update_dispersion(X, y, eta_n, disp_o, step_size)
        likelihood_n = family.negloglikelihood(y, eta_n, disp_n)
        diff = likelihood_n - likelihood_o

        return likelihood_n, diff, num_iter + 1, beta, eta_n, disp_n

    def cond_fun(val: Tuple):
        likelihood_o, diff, num_iter, beta, eta, disp = val
        del likelihood_o, beta, eta, disp
        return jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)

    init_beta = jnp.zeros((p,))
    init_eta = eta + offset_eta
    init_likelihood = family.negloglikelihood(y, init_eta, disp_init)
    init_tuple = (init_likelihood, jnp.inf, 0, init_beta, init_eta, disp_init)

    objective, objective_delta, num_iters, beta, eta, disp = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(objective_delta) < tol, num_iters <= max_iter)

    return _IRLSState(beta, num_iters, converged, disp, objective, objective_delta)


class IRLSFitter(Fitter, strict=True):
    """IRLS fit strategy implementing the `Fitter` contract."""

    def __call__(
        self,
        model: "GLM",
        data: GLMData,
        init: "Params | None" = None,
        max_iter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 1.0,
    ) -> "FitResult":
        if not isinstance(data, GLMData):
            raise TypeError("fit(...) expects `data` to be a GLMData instance.")
        if data.weights is not None:
            raise ValueError("GLMData.weights is not supported yet.")
        X, y, offset, _, _ = data.canonical_arrays()
        if X.shape[0] == 0:
            raise ValueError("GLMData.mask removes all samples; at least one effective sample is required.")

        init_beta, init_disp = _canonicalize_init(init, X.shape[1])
        if init_beta is not None:
            init_eta = jnp.asarray(X @ init_beta)
            if not bool(jnp.all(jnp.isfinite(init_eta))):
                raise ValueError("init_eta derived from init.beta must be finite.")
        else:
            init_eta = model.family.init_eta(y)

        if init_disp is not None:
            disp_init = jnp.asarray(init_disp)
            if not bool(jnp.all(jnp.isfinite(disp_init))):
                raise ValueError("disp_init must be finite.")
        else:
            disp_init = model.family.canonical_dispersion(1.0)

        state = irls(
            X,
            y,
            model.family,
            model.solver,
            init_eta,
            max_iter,
            tol,
            step_size,
            offset,
            disp_init=disp_init,
        )
        beta, n_iter, converged, irls_disp, objective, objective_delta = state

        eta = X @ beta + offset
        disp = model.family.estimate_dispersion(X, y, eta, irls_disp)
        mu = model.family.glink.inverse(eta)
        score_residual = (y - mu) * model.family.glink.deriv(mu)
        _, _, weight = model.family.calc_weight(eta, irls_disp)

        beta = jnp.ravel(beta)

        return FitResult(
            params=Params(beta=beta, disp=model.family.canonical_dispersion(disp)),
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
