import numpy as np

from jax import numpy as jnp
from jaxtyping import ArrayLike

from .family.dist import ExponentialFamily
from .glm import GLM, GLMState
from .infer.solve import AbstractLinearSolver
from .infer.stderr import AbstractStdErrEstimator


def _to_numeric_array(name: str, value: ArrayLike) -> jnp.ndarray:
    np_value = np.asarray(value)
    if np_value.dtype.kind not in ("i", "u", "f"):
        raise TypeError(f"{name} must have a numeric dtype")
    return jnp.asarray(value)


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
    if fitter is not None:
        raise TypeError("fitter strategy is not supported")
    if tests is not None:
        raise TypeError("tests strategy is not supported")
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
        model = GLM(family=model.family, solver=solver)

    fit_options = options or {}
    if covariance is not None:
        fit_options["se_estimator"] = covariance
    return model.fit(X_arr, y_arr, offset_eta=offset_arr, init=init, **fit_options)
