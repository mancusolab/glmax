# pattern: Functional Core

"""Fisher scoring Newton optimizer kernel and fitter strategy."""

from typing import cast, NamedTuple

import jax.numpy as jnp
import lineax as lx

from jax import Array, lax

from ..family import ExponentialDispersionFamily
from .types import AbstractFitter, FitResult, Params


__all__ = ["NewtonFitter"]


class _NewtonState(NamedTuple):
    beta: Array
    num_iters: int
    converged: Array
    disp: Array
    aux: Array
    objective: Array
    objective_delta: Array


def _newton(
    X: Array,
    y: Array,
    family: ExponentialDispersionFamily,
    solver: lx.AbstractLinearSolver,
    eta: Array,
    offset_eta: Array,
    disp_init: Array,
    aux_init: Array | None,
    max_iter: int = 200,
    tol: float = 1e-6,
    step_size: float = 1.0,
    armijo_c: float = 0.1,
    armijo_factor: float = 0.5,
    armijo_max_steps: int = 30,
) -> _NewtonState:
    """Fisher scoring Newton to solve a GLM."""
    _, p = X.shape
    # Xop = lx.MatrixLinearOperator(X)
    if not isinstance(solver, (lx.QR, lx.SVD)):
        solver = lx.Normal(solver)
    step_size = cast(Array, jnp.asarray(step_size))

    def body_fun(val: tuple[Array, ...]):
        likelihood_o, diff, num_iter, beta_o, eta_o, disp_o, aux_o = val

        # Compute GLM weights and residuals.
        mu_k, g_deriv_k, weight_k = family.calc_weight(eta_o, disp_o, aux_o)

        # Newton direction: solve (X^T W X) delta = X^T [w * g'(mu) * (mu - y)].
        # Expressed as weighted normal equations: (W^{1/2} X)^T (W^{1/2} X) delta
        #   = (W^{1/2} X)^T (W^{1/2} r) where r = g'(mu) * (mu - y).
        r = g_deriv_k * (mu_k - y)

        # TODO: lineax has a bug atm that converts Woh into n x n matrix which dramatically slows things down
        # workaround is the below fix. will stay in place until addressed in lineax.
        # Wh = lx.DiagonalLinearOperator(jnp.sqrt(weight_k))
        # A = Wh @ Xop
        # b = Wh.mv(r)
        sqrt_w = jnp.sqrt(weight_k)
        A = lx.MatrixLinearOperator(X * sqrt_w[:, jnp.newaxis])
        b = r * sqrt_w
        delta_beta = lx.linear_solve(A, b, solver=solver).value

        # Directional derivative gradient^T delta for the Armijo condition.
        # gradient = X^T (w * r) = (X^T W X) delta, so this is a PSD quadratic form.
        gradient = X.T @ (weight_k * r)
        grad_dot_delta = jnp.dot(gradient, delta_beta)

        # Backtracking Armijo line search.
        # Condition: f(beta - s * delta) <= f(beta) - c * s * gradient^T delta.
        def armijo_cond(ls_val: tuple) -> Array:
            s, obj, step = ls_val
            return jnp.logical_and(
                obj > likelihood_o - armijo_c * s * grad_dot_delta,
                step < armijo_max_steps,
            )

        def armijo_body(ls_val: tuple) -> tuple:
            s, _, step = ls_val
            s_new = s * armijo_factor
            beta_try = beta_o - s_new * delta_beta
            eta_try = X @ beta_try + offset_eta
            obj_try = family.negloglikelihood(y, eta_try, disp_o, aux_o)
            return s_new, obj_try, step + 1

        s0 = step_size
        obj0 = family.negloglikelihood(y, X @ (beta_o - s0 * delta_beta) + offset_eta, disp_o, aux_o)
        s, likelihood_n, _ = lax.while_loop(armijo_cond, armijo_body, (s0, obj0, 0))

        beta_n = beta_o - s * delta_beta
        eta_n = X @ beta_n + offset_eta

        disp_n, aux_n = family.update_nuisance(X, y, eta_n, disp_o, s, aux=aux_o)
        diff_n = likelihood_n - likelihood_o

        return likelihood_n, diff_n, num_iter + 1, beta_n, eta_n, disp_n, aux_n

    def cond_fun(val: tuple) -> Array:
        _, diff, num_iter, *_ = val
        return jnp.logical_and(jnp.fabs(diff) > tol, num_iter <= max_iter)

    init_beta = jnp.zeros((p,))
    init_eta_ = eta + offset_eta
    disp_init_ = jnp.asarray(disp_init)
    init_likelihood = family.negloglikelihood(y, init_eta_, disp_init_, aux_init)
    init_tuple = (init_likelihood, jnp.inf, 0, init_beta, init_eta_, disp_init_, aux_init)

    objective, objective_delta, num_iters, beta, _, disp, aux = lax.while_loop(cond_fun, body_fun, init_tuple)
    converged = jnp.logical_and(jnp.fabs(objective_delta) < tol, num_iters <= max_iter)

    return _NewtonState(beta, num_iters, converged, disp, aux, objective, objective_delta)


class NewtonFitter(AbstractFitter, strict=True):
    r"""Fisher scoring Newton fit strategy with backtracking Armijo line search.

    An [`glmax.AbstractFitter`][] that solves the GLM log-likelihood directly
    using Newton's method with the Fisher information matrix as the Hessian.
    Each Newton step solves

    $$\Delta\beta = (X^\top W X)^{-1} X^\top [w \odot g'(\mu) \odot (\mu - y)]$$

    where $W = \text{diag}(w)$ are the GLM working weights from
    [`glmax.ExponentialDispersionFamily.calc_weight`][]. Step size is chosen by
    backtracking Armijo: starting from `step_size` and shrinking by
    `armijo_factor` until the sufficient-decrease condition holds.

    Compared to [`glmax.IRLSFitter`][], `NewtonFitter` takes fewer outer
    iterations on problems where the IRLS fixed-step update overshoots — in
    particular with non-canonical links or near-boundary means.
    """

    solver: lx.AbstractLinearSolver
    step_size: float
    tol: float
    max_iter: int
    armijo_c: float
    armijo_factor: float

    def __init__(
        self,
        solver: lx.AbstractLinearSolver = lx.Cholesky(),
        step_size: float = 1.0,
        tol: float = 1e-6,
        max_iter: int = 200,
        armijo_c: float = 0.1,
        armijo_factor: float = 0.5,
    ):
        r"""Construct a Newton fitter.

        **Arguments:**

        - `solver`: `lineax` solver for each Newton weighted normal-equations
          step. Defaults to `lx.Cholesky()`. Any `lx.AbstractLinearSolver`
          that handles symmetric positive-semidefinite systems works here.
        - `step_size`: initial trial step size for the Armijo line search.
          Defaults to `1.0` (pure Newton step). Shrunk geometrically by
          `armijo_factor` until sufficient decrease is satisfied.
        - `tol`: convergence tolerance on the objective change between
          consecutive iterations. Defaults to `1e-6`.
        - `max_iter`: maximum number of Newton iterations. Defaults to `200`.
        - `armijo_c`: sufficient-decrease constant $c$ in the Armijo condition
          $f(\beta - s\Delta\beta) \leq f(\beta) - cs \cdot \nabla f^\top
          \Delta\beta$. Defaults to `0.1`.
        - `armijo_factor`: geometric backtracking multiplier. Each rejected
          trial step is shrunk by this factor. Defaults to `0.5`.
        """
        self.solver = solver
        self.step_size = step_size
        self.tol = tol
        self.max_iter = max_iter
        self.armijo_c = armijo_c
        self.armijo_factor = armijo_factor

    def fit(
        self,
        family: ExponentialDispersionFamily,
        X: Array,
        y: Array,
        offset: Array,
        weights: Array | None,
        init: Params | None = None,
    ) -> FitResult:
        r"""Run Fisher scoring Newton to convergence and return a `FitResult`.

        Each iteration computes the full Newton step using the Fisher
        information matrix, then uses backtracking Armijo to find a step size
        that guarantees sufficient decrease in the negative log-likelihood.

        **Arguments:**

        - `family`: [`glmax.ExponentialDispersionFamily`][] instance.
        - `X`: covariate matrix, shape `(n, p)`.
        - `y`: response vector, shape `(n,)`.
        - `offset`: offset vector, shape `(n,)`.
        - `weights`: per-sample weights; raises `ValueError` until implemented.
        - `init`: optional [`glmax.Params`][] for warm-starting; `None` uses
          the family default.

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

        newton_state = _newton(
            X,
            y,
            family,
            self.solver,
            init_eta,
            offset,
            disp_init,
            aux_init,
            self.max_iter,
            self.tol,
            self.step_size,
            self.armijo_c,
            self.armijo_factor,
        )
        beta, n_iter, converged, disp, aux, objective, objective_delta = newton_state

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
