# pattern: Functional Core

from typing import Tuple

from jax import Array, numpy as jnp
from jaxtyping import ArrayLike, ScalarLike

from .family.dist import ExponentialFamily, Gaussian, NegativeBinomial, Poisson
from .glm import GLMState
from .infer.contracts import AbstractLinearSolver
from .infer.fitters import AbstractGLMFitter, IRLSFitter
from .infer.inference import (
    AbstractStdErrEstimator,
    FisherInfoError,
    wald_test as inference_wald_test,
)
from .infer.solvers import CholeskySolver


def _to_numeric_array(name: str, value: ArrayLike) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.number):
        raise TypeError(f"{name} must have a numeric dtype")
    return array


def _normalize_fit_inputs(
    X: ArrayLike,
    y: ArrayLike,
    offset_eta: ArrayLike = 0.0,
    init: ArrayLike = None,
    alpha_init: ScalarLike = None,
) -> Tuple[Array, Array, Array, Array | None, Array | None]:
    X_array = _to_numeric_array("X", X)
    y_array = _to_numeric_array("y", y)
    offset_array = _to_numeric_array("offset_eta", offset_eta)
    init_array = None if init is None else _to_numeric_array("init", init)
    alpha_array = None if alpha_init is None else _to_numeric_array("alpha_init", alpha_init)

    return X_array, y_array, offset_array, init_array, alpha_array


def _calc_eta_and_dispersion(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    solver: AbstractLinearSolver,
    se_estimator: AbstractStdErrEstimator,
    offset_eta: ArrayLike = 0.0,
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
) -> Tuple[Array, Array]:
    n = X.shape[0]
    init_val = family.init_eta(y)

    if isinstance(family, NegativeBinomial):
        glm_state_pois = fit(
            X,
            y,
            family=Poisson(),
            solver=solver,
            offset_eta=offset_eta,
            init=init_val,
            alpha_init=jnp.asarray(0.0),
            se_estimator=se_estimator,
            max_iter=max_iter,
            tol=tol,
            step_size=step_size,
        )

        alpha_init = n / jnp.sum((y / family.glink.inverse(glm_state_pois.eta) - 1) ** 2)
        eta = glm_state_pois.eta
        disp = family.estimate_dispersion(X, y, eta, alpha=1.0 / alpha_init, max_iter=max_iter)
        disp = jnp.nan_to_num(disp, nan=0.1)
    else:
        eta = init_val
        disp = jnp.asarray(0.0)

    return eta, disp


def fit(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily = Gaussian(),
    solver: AbstractLinearSolver = CholeskySolver(),
    fitter: AbstractGLMFitter = IRLSFitter(),
    offset_eta: ArrayLike = 0.0,
    init: ArrayLike = None,
    alpha_init: ScalarLike = None,
    se_estimator: AbstractStdErrEstimator = FisherInfoError(),
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
) -> GLMState:
    if not isinstance(fitter, AbstractGLMFitter):
        raise TypeError("fitter must implement AbstractGLMFitter")
    X, y, offset_eta, init, alpha_init = _normalize_fit_inputs(X, y, offset_eta, init, alpha_init)

    if init is None or alpha_init is None:
        init, alpha_init = _calc_eta_and_dispersion(
            X,
            y,
            family=family,
            solver=solver,
            se_estimator=se_estimator,
            offset_eta=offset_eta,
            max_iter=max_iter,
            tol=tol,
            step_size=step_size,
        )

    fit_state = fitter(
        X,
        y,
        family,
        solver,
        init,
        max_iter=max_iter,
        tol=tol,
        step_size=step_size,
        offset_eta=offset_eta,
        alpha_init=alpha_init,
    )
    beta = fit_state.beta
    n_iter = fit_state.num_iters
    converged = fit_state.converged
    alpha = fit_state.alpha

    eta = X @ beta + offset_eta
    mu = family.glink.inverse(eta)
    resid = (y - mu) * family.glink.deriv(mu)

    _, _, weight = family.calc_weight(X, y, eta, alpha)

    resid_covar = se_estimator(family, X, y, eta, mu, weight, alpha)
    beta_se = jnp.sqrt(jnp.diag(resid_covar))

    df = X.shape[0] - X.shape[1]
    beta = beta.squeeze()
    stat = beta / beta_se

    pval_wald = inference_wald_test(stat, df, family)

    return GLMState(
        beta,
        beta_se,
        stat,
        pval_wald,
        eta,
        mu,
        weight,
        n_iter,
        converged,
        resid_covar,
        resid,
        alpha,
    )
