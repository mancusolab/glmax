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


def _validated_params(params: Params, n_features: int) -> Params:
    beta, disp, aux = _canonicalize_init(params, n_features)
    assert beta is not None and disp is not None
    return Params(beta=beta, disp=disp, aux=aux)


def _canonicalize_model_params(model: GLM, params: Params) -> Params:
    canonical_disp, canonical_aux = model.canonicalize_params(params.disp, params.aux)
    return Params(beta=params.beta, disp=canonical_disp, aux=canonical_aux)


@eqx.filter_jit
def fit(model: GLM, data: GLMData, init: Params | None = None, *, fitter: AbstractFitter = IRLSFitter()) -> FittedGLM:
    r"""Fit a GLM to observed data and return a fitted noun.

    This is the canonical `fit` grammar verb. It is `@eqx.filter_jit`-wrapped;
    the fitter strategy is treated as static structure under JIT.

    **Arguments:**

    - `model`: `GLM` specification noun produced by `specify(...)`.
    - `data`: `GLMData` noun carrying `X` and `y` (and optional `offset`).
    - `init`: optional `Params` for warm-starting; `None` uses the family default.
    - `fitter`: `AbstractFitter` strategy (default: `IRLSFitter()`).

    **Returns:**

    `FittedGLM` noun binding the model and the `FitResult` contract.

    **Raises:**

    - `TypeError`: if `model`, `data`, `init`, or `fitter` have wrong types,
      or if the fitter does not return a `FitResult`.
    - `ValueError`: if `data.weights` is set (not yet supported).
    """

    if not isinstance(model, GLM):
        raise TypeError("fit(...) expects `model` to be a GLM instance.")
    if not isinstance(data, GLMData):
        raise TypeError("fit(...) expects `data` to be a GLMData instance.")
    if init is not None and not isinstance(init, Params):
        raise TypeError("fit(...) expects `init` to be a Params instance or None.")
    if not isinstance(fitter, AbstractFitter):
        raise TypeError("fit(...) expects `fitter` to be an AbstractFitter instance.")
    if init is not None:
        init = _validated_params(init, data.X.shape[1])
        if not isinstance(fitter, IRLSFitter):
            init = _canonicalize_model_params(model, init)

    result = fitter(model, data, init)

    if not isinstance(result, FitResult):
        raise TypeError("fit(...) expects `fitter` to return a FitResult instance.")

    return FittedGLM(model=model, result=result)


@eqx.filter_jit
def predict(model: GLM, params: Params, data: GLMData) -> jnp.ndarray:
    r"""Apply a fitted model to new data and return predicted means.

    This is the canonical `predict` grammar verb. It is `@eqx.filter_jit`-wrapped.

    **Arguments:**

    - `model`: `GLM` specification noun.
    - `params`: fitted `Params` (e.g. `fitted.params` from `fit(...)`).
    - `data`: `GLMData` noun carrying covariates `X` (and optional `offset`).

    **Returns:**

    Predicted mean response $\hat\mu = g^{-1}(X\hat\beta + \text{offset})$, shape `(n,)`.

    **Raises:**

    - `TypeError`: if `model`, `params`, or `data` have wrong types.
    - `ValueError`: if `data.weights` is set (not yet supported).
    """

    if not isinstance(model, GLM):
        raise TypeError("predict(...) expects `model` to be a GLM instance.")
    if not isinstance(params, Params):
        raise TypeError("predict(...) expects `params` to be a Params instance.")
    if not isinstance(data, GLMData):
        raise TypeError("predict(...) expects `data` to be a GLMData instance.")
    if data.weights is not None:
        raise ValueError("GLMData.weights is not supported in predict yet.")

    X_array, _, offset_array, _ = data.canonical_arrays()
    params = _canonicalize_model_params(model, _validated_params(params, X_array.shape[1]))

    eta = X_array @ params.beta + offset_array
    return model.mean(eta)
