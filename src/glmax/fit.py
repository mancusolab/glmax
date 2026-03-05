# pattern: Functional Core

from __future__ import annotations

import jax.numpy as jnp

from .contracts import FitResult, Fitter, GLMData, Params, validate_fit_result
from .glm import _fit_model, GLM


def _canonicalize_init(init: Params | None, n_features: int) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    if init is None:
        return None, None

    try:
        beta = jnp.asarray(init.beta)
    except TypeError as exc:
        raise TypeError("Params.beta must be numeric.") from exc
    if not jnp.issubdtype(beta.dtype, jnp.number):
        raise TypeError("Params.beta must be numeric.")
    if beta.ndim != 1 or beta.shape[0] != n_features:
        raise ValueError("Params.beta must be a one-dimensional vector with length equal to X.shape[1].")
    if not bool(jnp.issubdtype(beta.dtype, jnp.inexact)):
        raise TypeError("Params.beta must have an inexact dtype.")
    if not bool(jnp.all(jnp.isfinite(beta))):
        raise ValueError("Params.beta must contain only finite values.")

    try:
        disp = jnp.asarray(init.disp)
    except TypeError as exc:
        raise TypeError("Params.disp must be numeric.") from exc
    if not jnp.issubdtype(disp.dtype, jnp.number):
        raise TypeError("Params.disp must be numeric.")
    if disp.ndim > 0 and disp.size != 1:
        raise ValueError("Params.disp must be a scalar.")
    if not bool(jnp.issubdtype(disp.dtype, jnp.inexact)):
        raise TypeError("Params.disp must have an inexact dtype.")
    if not bool(jnp.all(jnp.isfinite(disp))):
        raise ValueError("Params.disp must contain only finite values.")

    return beta, disp


class _ModelFitter:
    """Bridge canonical fit verb calls into the canonical GLM fit kernel."""

    def __call__(self, model: GLM, data: GLMData, init: Params | None = None) -> FitResult:
        X_array, _, offset_array, _, _ = data.canonical_arrays()
        init_beta, init_disp = _canonicalize_init(init, X_array.shape[1])
        init_eta = None if init_beta is None else X_array @ init_beta + offset_array
        return _fit_model(model, data, init_eta=init_eta, disp_init=init_disp)


DEFAULT_FITTER: Fitter = _ModelFitter()


def fit(model: GLM, data: GLMData, init: Params | None = None, *, fitter: Fitter = DEFAULT_FITTER) -> FitResult:
    """Canonical public fit verb surface over grammar nouns."""
    if not isinstance(model, GLM):
        raise TypeError("fit(...) expects `model` to be a GLM instance.")
    if not isinstance(data, GLMData):
        raise TypeError("fit(...) expects `data` to be a GLMData instance.")
    if init is not None and not isinstance(init, Params):
        raise TypeError("fit(...) expects `init` to be a Params instance or None.")

    result = fitter(model, data, init)
    if not isinstance(result, FitResult):
        raise TypeError("fit(...) expects `fitter` to return a FitResult instance.")
    validate_fit_result(result)
    return result


def predict(model: GLM, params: Params, data: GLMData) -> jnp.ndarray:
    """Pure prediction verb over grammar nouns."""
    if not isinstance(model, GLM):
        raise TypeError("predict(...) expects `model` to be a GLM instance.")
    if not isinstance(params, Params):
        raise TypeError("predict(...) expects `params` to be a Params instance.")
    if not isinstance(data, GLMData):
        raise TypeError("predict(...) expects `data` to be a GLMData instance.")
    if data.weights is not None:
        raise ValueError("GLMData.weights is not supported in predict yet.")

    X_array, _, offset_array, _, _ = data.canonical_arrays()
    beta, disp = _canonicalize_init(params, X_array.shape[1])
    assert beta is not None and disp is not None

    eta = X_array @ beta + offset_array
    return model.family.glink.inverse(eta)
