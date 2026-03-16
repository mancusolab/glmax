# pattern: Functional Core

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ..data import GLMData
from ..glm import GLM
from .irls import IRLSFitter
from .types import (
    _canonicalize_init,
    AbstractFitter,
    FitResult,
    FittedGLM,
    Params,
)


__all__ = ["fit", "predict"]


@eqx.filter_jit
def fit(model: GLM, data: GLMData, init: Params | None = None, *, fitter: AbstractFitter = IRLSFitter()) -> FittedGLM:
    """Canonical public fit verb over grammar nouns."""

    if not isinstance(model, GLM):
        raise TypeError("fit(...) expects `model` to be a GLM instance.")
    if not isinstance(data, GLMData):
        raise TypeError("fit(...) expects `data` to be a GLMData instance.")
    if init is not None and not isinstance(init, Params):
        raise TypeError("fit(...) expects `init` to be a Params instance or None.")
    if not isinstance(fitter, AbstractFitter):
        raise TypeError("fit(...) expects `fitter` to be an AbstractFitter instance.")

    result = fitter(model, data, init)

    if not isinstance(result, FitResult):
        raise TypeError("fit(...) expects `fitter` to return a FitResult instance.")

    return FittedGLM(model=model, result=result)


@eqx.filter_jit
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

    X_array, _, offset_array, _ = data.canonical_arrays()
    beta, disp = _canonicalize_init(params, X_array.shape[1])
    assert beta is not None and disp is not None

    eta = X_array @ beta + offset_array
    return model.family.glink.inverse(eta)
