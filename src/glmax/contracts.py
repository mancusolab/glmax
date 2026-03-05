from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, TYPE_CHECKING

import jax.numpy as jnp

from jax import Array, tree_util
from jaxtyping import ArrayLike


if TYPE_CHECKING:
    from .glm import GLM


def _is_numeric_dtype(array: Array) -> bool:
    return bool(jnp.issubdtype(array.dtype, jnp.number))


def _as_numeric_array(name: str, value: ArrayLike) -> Array:
    array = jnp.asarray(value)
    if not _is_numeric_dtype(array):
        raise TypeError(f"GLMData.{name} must be numeric.")
    return array


def _require_finite(name: str, array: Array) -> None:
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"GLMData.{name} must contain only finite values.")


def _canonicalize_numeric_vector(name: str, value: ArrayLike, n_samples: int) -> Array:
    array = _as_numeric_array(name, value)
    if array.ndim == 0:
        vector = jnp.full((n_samples,), array)
    elif array.ndim == 1 and array.shape[0] == n_samples:
        vector = array
    else:
        raise ValueError(f"GLMData.{name} must be scalar-broadcastable or length n.")

    _require_finite(name, vector)
    return vector


def _canonicalize_mask(mask: ArrayLike, n_samples: int) -> Array:
    array = jnp.asarray(mask)
    if not jnp.issubdtype(array.dtype, jnp.bool_):
        raise TypeError("GLMData.mask must be boolean.")

    if array.ndim == 0:
        return jnp.full((n_samples,), array, dtype=jnp.bool_)
    if array.ndim == 1 and array.shape[0] == n_samples:
        return array

    raise ValueError("GLMData.mask must be scalar-broadcastable or length n.")


@tree_util.register_dataclass
@dataclass(frozen=True)
class GLMData:
    """Canonical data noun for GLM workflows."""

    X: ArrayLike
    y: ArrayLike
    offset: ArrayLike | None = None
    weights: ArrayLike | None = None
    mask: ArrayLike | None = None

    def __post_init__(self) -> None:
        X = _as_numeric_array("X", self.X)
        y = _as_numeric_array("y", self.y)

        if X.ndim != 2:
            raise ValueError("GLMData.X must be rank-2 with shape (n, p).")
        if y.ndim != 1:
            raise ValueError("GLMData.y must be rank-1 with shape (n,).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("GLMData.X and GLMData.y must share the sample dimension n.")

        _require_finite("X", X)
        _require_finite("y", y)

        n_samples = X.shape[0]
        offset = None
        if self.offset is not None:
            offset = _canonicalize_numeric_vector("offset", self.offset, n_samples)

        weights = None
        if self.weights is not None:
            weights = _canonicalize_numeric_vector("weights", self.weights, n_samples)

        mask = None
        if self.mask is not None:
            mask = _canonicalize_mask(self.mask, n_samples)

        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "mask", mask)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def canonical_offset(self) -> Array:
        if self.offset is None:
            return jnp.zeros((self.n_samples,), dtype=self.y.dtype)
        return self.offset

    def canonical_weights(self) -> Array:
        if self.weights is None:
            dtype = jnp.result_type(self.X.dtype, self.y.dtype, jnp.float32)
            return jnp.ones((self.n_samples,), dtype=dtype)
        return self.weights

    def canonical_mask(self) -> Array:
        if self.mask is None:
            return jnp.ones((self.n_samples,), dtype=jnp.bool_)
        return self.mask

    def canonical_arrays(self) -> tuple[Array, Array, Array, Array, Array]:
        mask = self.canonical_mask()
        return (
            self.X[mask],
            self.y[mask],
            self.canonical_offset()[mask],
            self.canonical_weights()[mask],
            mask,
        )


@tree_util.register_dataclass
@dataclass(frozen=True)
class Params:
    """Canonical model parameters."""

    beta: Array
    disp: Array


@tree_util.register_dataclass
@dataclass(frozen=True)
class Diagnostics:
    """Common diagnostics emitted by fit/check verbs."""

    converged: Array
    num_iters: Array


@tree_util.register_dataclass
@dataclass(frozen=True)
class FitResult:
    """Canonical fit contract shared by grammar verbs."""

    params: Params
    se: Array
    z: Array
    p: Array
    eta: Array
    mu: Array
    glm_wt: Array
    diagnostics: Diagnostics
    infor_inv: Array
    resid: Array

    @property
    def beta(self) -> Array:
        return self.params.beta

    @property
    def alpha(self) -> Array:
        return self.params.disp

    @property
    def num_iters(self) -> Array:
        return self.diagnostics.num_iters

    @property
    def converged(self) -> Array:
        return self.diagnostics.converged


@tree_util.register_dataclass
@dataclass(frozen=True)
class InferenceResult:
    """Canonical infer verb output contract."""

    params: Params
    se: Array
    z: Array
    p: Array


@runtime_checkable
class Fitter(Protocol):
    """Canonical fitter protocol consumed by the top-level fit verb."""

    def __call__(self, model: GLM, data: GLMData, init: Params | None = None) -> FitResult:
        ...
