from __future__ import annotations

import jax.numpy as jnp

from .contracts import FitResult, Fitter, GLMData, Params
from .glm import GLM


def _canonicalize_offset(offset: object, n_samples: int):
    if offset is None:
        return 0.0

    offset_array = jnp.asarray(offset)
    if offset_array.ndim == 0:
        return offset_array
    if offset_array.ndim == 1 and offset_array.shape[0] == n_samples:
        return offset_array
    raise ValueError("GLMData.offset must be a scalar or a length-n vector.")


def _canonicalize_init(init: Params | None, n_features: int) -> tuple[jnp.ndarray | None, jnp.ndarray | None]:
    if init is None:
        return None, None

    beta = jnp.asarray(init.beta)
    if beta.ndim != 1 or beta.shape[0] != n_features:
        raise ValueError("Params.beta must be a one-dimensional vector with length equal to X.shape[1].")

    disp = jnp.asarray(init.disp)
    if disp.ndim > 0 and disp.size != 1:
        raise ValueError("Params.disp must be a scalar.")

    return beta, disp


def _validate_and_canonicalize(data: GLMData, init: Params | None):
    if data.weights is not None:
        raise ValueError("GLMData.weights is not supported by the default fitter yet.")
    if data.mask is not None:
        raise ValueError("GLMData.mask is not supported by the default fitter yet.")

    X = jnp.asarray(data.X)
    y = jnp.asarray(data.y)

    if X.ndim != 2:
        raise ValueError("GLMData.X must be a two-dimensional array with shape (n, p).")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("GLMData.y must be a one-dimensional array with length equal to X.shape[0].")

    offset = _canonicalize_offset(data.offset, X.shape[0])
    init_beta, init_disp = _canonicalize_init(init, X.shape[1])
    return X, y, offset, init_beta, init_disp


class _ModelFitter:
    """Bridge canonical fit verb calls into the existing GLM.fit implementation."""

    def __call__(self, model: GLM, data: GLMData, init: Params | None = None) -> FitResult:
        X, y, offset, init_beta, init_disp = _validate_and_canonicalize(data, init)
        if init_beta is None:
            return model.fit(X, y, offset_eta=offset)

        eta_init = X @ init_beta
        return model.fit(X, y, offset_eta=offset, init=eta_init, alpha_init=init_disp)


DEFAULT_FITTER: Fitter = _ModelFitter()


def fit(model: GLM, data: GLMData, init: Params | None = None, *, fitter: Fitter = DEFAULT_FITTER) -> FitResult:
    """Canonical fit verb surface."""
    return fitter(model, data, init)
