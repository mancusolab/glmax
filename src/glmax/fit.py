import numpy as np

import equinox as eqx

from jax import numpy as jnp
from jaxtyping import ArrayLike

from .family.dist import ExponentialFamily
from .glm import GLM, GLMState
from .infer.fitter import AbstractFitter
from .infer.optimize import irls
from .infer.solve import AbstractLinearSolver
from .infer.stderr import AbstractStdErrEstimator, FisherInfoError
from .infer.tests import AbstractHypothesisTest, WaldTest


def _to_numeric_array(name: str, value: ArrayLike) -> jnp.ndarray:
    np_value = np.asarray(value)
    if np_value.dtype.kind not in ("i", "u", "f"):
        raise TypeError(f"{name} must have a numeric dtype")
    return jnp.asarray(value)


def _run_default_pipeline(
    model: GLM,
    X: jnp.ndarray,
    y: jnp.ndarray,
    offset: jnp.ndarray,
    *,
    init: ArrayLike | None,
    covariance: AbstractStdErrEstimator | None,
    tests: AbstractHypothesisTest | None,
    options: dict[str, object],
) -> GLMState:
    max_iter = int(options.pop("max_iter", 1000))
    tol = float(options.pop("tol", 1e-3))
    step_size = float(options.pop("step_size", 1.0))
    alpha_init = options.pop("alpha_init", None)
    if options:
        unknown_keys = ", ".join(sorted(options.keys()))
        raise TypeError(f"Unknown fit options: {unknown_keys}")

    if init is None and alpha_init is None:
        eta_init, alpha_init = model.calc_eta_and_dispersion(X, y, offset)
    elif init is None:
        eta_init, _ = model.calc_eta_and_dispersion(X, y, offset)
    elif alpha_init is None:
        eta_init = jnp.asarray(init)
        _, alpha_init = model.calc_eta_and_dispersion(X, y, offset)
    else:
        eta_init = jnp.asarray(init)

    beta, n_iter, converged, alpha = irls(
        X,
        y,
        model.family,
        model.solver,
        eta_init,
        max_iter,
        tol,
        step_size,
        offset,
        alpha_init,
    )

    eta = X @ beta + offset
    mu = model.family.glink.inverse(eta)
    resid = (y - mu) * model.family.glink.deriv(mu)
    _, _, weight = model.family.calc_weight(X, y, eta, alpha)

    se_estimator = FisherInfoError() if covariance is None else covariance
    resid_covar = se_estimator(model.family, X, y, eta, mu, weight, alpha)
    beta_se = jnp.sqrt(jnp.diag(resid_covar))

    df = X.shape[0] - X.shape[1]
    beta = beta.squeeze()
    stat = beta / beta_se

    hypothesis_test = WaldTest() if tests is None else tests
    pval = hypothesis_test(stat, df, model.family)

    return GLMState(
        beta,
        beta_se,
        stat,
        pval,
        eta,
        mu,
        weight,
        n_iter,
        converged,
        resid_covar,
        resid,
        alpha,
    )


def fit(
    model: GLM,
    X: ArrayLike,
    y: ArrayLike,
    offset: ArrayLike | None = None,
    *,
    fitter: object | None = None,
    solver: object | None = None,
    covariance: object | None = None,
    tests: object | None = None,
    init: ArrayLike | None = None,
    options: dict[str, object] | None = None,
) -> GLMState:
    if fitter is not None and not isinstance(fitter, AbstractFitter):
        raise TypeError("fitter must implement AbstractFitter")
    if tests is not None and not isinstance(tests, AbstractHypothesisTest):
        raise TypeError("tests must implement AbstractHypothesisTest")
    if solver is not None and not isinstance(solver, AbstractLinearSolver):
        raise TypeError("solver must implement AbstractLinearSolver")
    if covariance is not None and not isinstance(covariance, AbstractStdErrEstimator):
        raise TypeError("covariance must implement AbstractStdErrEstimator")
    if options is not None and not isinstance(options, dict):
        raise TypeError("options must be a dictionary")

    X_arr = _to_numeric_array("X", X)
    y_arr = _to_numeric_array("y", y)

    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    if offset is None:
        offset_arr = jnp.zeros((X_arr.shape[0],), dtype=y_arr.dtype)
    else:
        offset_arr = _to_numeric_array("offset", offset)
        if offset_arr.ndim != 1:
            raise ValueError("offset must be a 1D array")
        if offset_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("offset must have length equal to the number of rows in X")

    if not bool(jnp.all(jnp.isfinite(X_arr))):
        raise ValueError("X must contain only finite values")
    if not bool(jnp.all(jnp.isfinite(y_arr))):
        raise ValueError("y must contain only finite values")
    if not bool(jnp.all(jnp.isfinite(offset_arr))):
        raise ValueError("offset must contain only finite values")

    if isinstance(model.family, ExponentialFamily):
        try:
            model.family.__check_init__()
        except ValueError as exc:
            raise ValueError("Invalid family/link combination") from exc

    if solver is not None:
        model = eqx.tree_at(lambda m: m.solver, model, solver)

    fit_options = {} if options is None else dict(options)
    option_covariance = fit_options.pop("se_estimator", None)
    if option_covariance is not None and covariance is not None:
        raise ValueError("Specify covariance either via `covariance` or `options['se_estimator']`, not both")
    if covariance is None and option_covariance is not None:
        if not isinstance(option_covariance, AbstractStdErrEstimator):
            raise TypeError("options['se_estimator'] must implement AbstractStdErrEstimator")
        covariance = option_covariance

    if fitter is not None:
        if covariance is not None:
            fit_options["se_estimator"] = covariance
        if tests is not None:
            fit_options["test_hook"] = tests
        return fitter(model, X_arr, y_arr, offset_arr, init=init, options=fit_options)

    return _run_default_pipeline(
        model,
        X_arr,
        y_arr,
        offset_arr,
        init=init,
        covariance=covariance,
        tests=tests,
        options=fit_options,
    )
