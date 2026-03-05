"""Internal IRLS optimizer kernels used by canonical grammar verbs."""

from typing import NamedTuple, Tuple

from jax import Array, lax, numpy as jnp
from jaxtyping import ArrayLike, ScalarLike

from ..family.dist import ExponentialFamily
from .solve import AbstractLinearSolver


class _IRLSState(NamedTuple):
    beta: Array
    num_iters: int
    converged: Array
    disp: Array
    objective: Array
    objective_delta: Array


# @eqx.filter_jit
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
    """IRLS to solve GLM

    :param X: covariate data matrix (nxp)
    :param y: outcome vector (nx1)
    :param family: GLM model for running eQTL mapping, eg. Negative Binomial, Poisson
    :param solver: linear equation solver
    :param eta: linear component eta
    :param max_iter: maximum iterations for fitting GLM, default to 1000
    :param tol: tolerance for stopping, default to 0.001
    :param step_size: step size to update the parameter at each step, default to 1.0
    :param offset_eta: offset (nx1)
    :param disp_init: initial value for the canonical dispersion parameter
    :return: _IRLSState
    """
    n, p = X.shape

    def body_fun(val: Tuple):
        likelihood_o, diff, num_iter, beta_o, eta_o, disp_o = val

        mu_k, g_deriv_k, weight = family.calc_weight(X, y, eta_o, disp_o)
        r = eta_o + g_deriv_k * (y - mu_k) * step_size - offset_eta

        beta = solver(X, r, weight)

        eta_n = X @ beta + offset_eta

        disp_n = family.update_dispersion(X, y, eta_n, disp_o, step_size)
        likelihood_n = family.negloglikelihood(X, y, eta_n, disp_n)
        diff = likelihood_n - likelihood_o

        return likelihood_n, diff, num_iter + 1, beta, eta_n, disp_n

    def cond_fun(val: Tuple):
        likelihood_o, diff, num_iter, beta, eta, disp = val
        cond_l = jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)
        return cond_l

    init_beta = jnp.zeros((p,))
    init_eta = eta + offset_eta
    init_likelihood = family.negloglikelihood(X, y, init_eta, disp_init)
    init_tuple = (init_likelihood, jnp.inf, 0, init_beta, init_eta, disp_init)

    objective, objective_delta, num_iters, beta, eta, disp = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(objective_delta) < tol, num_iters <= max_iter)

    return _IRLSState(beta, num_iters, converged, disp, objective, objective_delta)
