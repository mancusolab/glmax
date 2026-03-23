# pattern: Functional Core

"""IRLS optimizer kernels and default fitter strategy."""

from typing import cast, NamedTuple

import jax.numpy as jnp
import lineax as lx

from jax import Array, lax
from jaxtyping import ArrayLike, ScalarLike

from ..family import ExponentialDispersionFamily
from .types import AbstractFitter, FitResult, Params


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
    X: Array,
    y: Array,
    family: ExponentialDispersionFamily,
    solver: lx.AbstractLinearSolver,
    eta: Array,
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
    offset_eta: ArrayLike = 0.0,
    disp_init: ScalarLike = 0.0,
    aux_init: ScalarLike = 0.0,
) -> _IRLSState:
    """IRLS to solve GLM."""
    _, p = X.shape

    Xop = lx.MatrixLinearOperator(X)
    step_size = cast(Array, jnp.asarray(step_size))
    if not isinstance(solver, (lx.QR, lx.SVD)):
        solver = lx.Normal(solver)

    def body_fun(val: tuple[Array, ...]):
        likelihood_o, diff, num_iter, _beta_o, eta_o, disp_o, aux_o = val

        # compute means, weights, and gradients
        mu_k, g_deriv_k, weight = family.calc_weight(eta_o, disp_o, aux_o)
        r = eta_o + g_deriv_k * (y - mu_k) * step_size - offset_eta

        # prepare for lineax and solve
        Woh = lx.DiagonalLinearOperator(jnp.sqrt(weight))
        A = Woh @ Xop
        b = Woh.mv(r)
        beta = lx.linear_solve(A, b, solver=solver).value

        # update eta
        eta_n = X @ beta + offset_eta

        # update any nuisance params (disp, overdisp)
        disp_n, aux_n = family.update_nuisance(X, y, eta_n, disp_o, step_size, aux=aux_o)

        # re-evaluate
        likelihood_n = family.negloglikelihood(y, eta_n, disp_n, aux_n)
        diff = likelihood_n - likelihood_o

        return likelihood_n, diff, num_iter + 1, beta, eta_n, disp_n, aux_n

    def cond_fun(val: tuple):
        likelihood_o, diff, num_iter, beta, eta, disp, aux = val
        del likelihood_o, beta, eta, disp, aux
        return jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)

    init_beta = jnp.zeros((p,))
    init_eta = eta + offset_eta
    init_likelihood = family.negloglikelihood(y, init_eta, disp_init, aux_init)
    init_tuple = (init_likelihood, jnp.inf, 0, init_beta, init_eta, disp_init, aux_init)

    objective, objective_delta, num_iters, beta, eta, disp, aux = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(objective_delta) < tol, num_iters <= max_iter)

    return _IRLSState(beta, num_iters, converged, disp, aux, objective, objective_delta)


class IRLSFitter(AbstractFitter, strict=True):
    r"""Iteratively Reweighted Least Squares (IRLS) fit strategy.

    The default [`glmax.AbstractFitter`][] used by [`glmax.fit`][]. At each
    iteration IRLS forms the adjusted response

    $$z_i = \eta_i + s \cdot g'(\mu_i)(y_i - \mu_i)$$

    and solves the weighted normal equations

    $$(X^\top W X)\,\hat\beta = X^\top W z$$

    where $W = \text{diag}(w)$ are the GLM working weights from
    [`glmax.ExponentialDispersionFamily.calc_weight`][] and $s$ is
    `step_size`. The linear system is solved with a `lineax` solver and the
    linear predictor is updated as $\eta \leftarrow X\hat\beta + \text{offset}$.

    IRLS is mathematically equivalent to Fisher scoring Newton but expressed as
    a sequence of weighted least-squares problems. It converges in one
    iteration for Gaussian/identity, where the working weights are constant and
    the objective is exactly quadratic. For all other families the weights
    depend on the current mean estimate, so multiple iterations are required.
    The fixed `step_size` controls the update magnitude; use
    [`glmax.NewtonFitter`][] if you want automatic backtracking line search.
    """

    solver: lx.AbstractLinearSolver
    step_size: float
    tol: float
    max_iter: int

    def __init__(
        self,
        solver: lx.AbstractLinearSolver = lx.Cholesky(),
        step_size: float = 1.0,
        tol: float = 1e-3,
        max_iter: int = 1000,
    ):
        r"""Construct an IRLS fitter.

        **Arguments:**

        - `solver`: `lineax` solver used for each IRLS weighted least-squares
          step. Defaults to `lx.Cholesky()`. Any `lx.AbstractLinearSolver`
          that handles symmetric positive-semidefinite systems works here.
        - `step_size`: IRLS update step-size multiplier. Defaults to `1.0`.
        - `tol`: convergence tolerance on the objective change. Defaults to `1e-3`.
        - `max_iter`: maximum number of IRLS iterations. Defaults to `1000`.
        """
        self.solver = solver
        self.step_size = step_size
        self.tol = tol
        self.max_iter = max_iter

    def fit(
        self,
        family: ExponentialDispersionFamily,
        X: Array,
        y: Array,
        offset: Array,
        weights: Array | None,
        init: Params | None = None,
    ) -> FitResult:
        r"""Run IRLS to convergence and return a `FitResult`.

        IRLS iterates on the linear predictor $\eta$ and nuisance state until
        the objective change falls below `self.tol` or the iteration count
        reaches `self.max_iter`.

        **Arguments:**

        - `family`: [`glmax.ExponentialDispersionFamily`][] instance.
        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: response vector, shape `(n,)`.
        - `offset`: offset vector, shape `(n,)`.
        - `weights`: optional per-sample weight vector, shape `(n,)`.
        - `init`: optional [`glmax.Params`][] for warm-starting; `None` uses
          the family default.
        - `step_size`: IRLS update step-size multiplier (default `1.0`).

        **Returns:**

        [`glmax.FitResult`][] with converged parameters, fit artifacts, and
        convergence metadata.
        """
        if weights is not None:
            raise ValueError("Per-sample weights are not supported yet.")

        default_disp, default_aux = family.init_nuisance()
        if init is not None:
            disp_init = jnp.asarray(init.disp)
            aux_init = jnp.asarray(init.aux) if init.aux is not None else default_aux
            init_eta = X @ jnp.asarray(init.beta) + offset
        else:
            disp_init = default_disp
            aux_init = default_aux
            init_eta = family.init_eta(y)

        irls_state = _irls(
            X,
            y,
            family,
            self.solver,
            init_eta,
            self.max_iter,
            self.tol,
            self.step_size,
            offset,
            disp_init=disp_init,
            aux_init=aux_init,
        )
        beta, n_iter, converged, disp, aux, objective, objective_delta = irls_state

        eta = X @ beta + offset
        mu, link_deriv, weight = family.calc_weight(eta, disp, aux)
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
