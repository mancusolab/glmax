# pattern: Functional Core

from __future__ import annotations

import jax.numpy as jnp

from ..data import GLMData
from ..glm import GLM
from .irls import IRLSFitter
from .types import (
    _canonicalize_init,
    _matches_fit_result_shape,
    _matches_fitter_shape,
    _matches_namedtuple_shape,
    FittedGLM,
    Fitter,
    Params,
    validate_fit_result,
)


__all__ = ["fit", "predict"]


def fit(model: GLM, data: GLMData, init: Params | None = None, *, fitter: Fitter = IRLSFitter()) -> FittedGLM:
    """Canonical public fit verb over grammar nouns."""

    if not isinstance(model, GLM):
        raise TypeError("fit(...) expects `model` to be a GLM instance.")
    if not isinstance(data, GLMData):
        raise TypeError("fit(...) expects `data` to be a GLMData instance.")
    if init is not None and not _matches_namedtuple_shape(init, type_name="Params", fields=("beta", "disp")):
        raise TypeError("fit(...) expects `init` to be a Params instance or None.")
    if not _matches_fitter_shape(fitter):
        raise TypeError("fit(...) expects `fitter` to be a Fitter instance.")

    result = fitter(model, data, init)
    if not _matches_fit_result_shape(result):
        raise TypeError("fit(...) expects `fitter` to return a FitResult instance.")
    validate_fit_result(result)

    return FittedGLM(model=model, result=result)


def predict(model: GLM, params: Params, data: GLMData) -> jnp.ndarray:
    """Pure prediction verb over grammar nouns."""

    if not isinstance(model, GLM):
        raise TypeError("predict(...) expects `model` to be a GLM instance.")
    if not _matches_namedtuple_shape(params, type_name="Params", fields=("beta", "disp")):
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
