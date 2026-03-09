# pattern: Functional Core

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jaxtyping import ArrayLike


__all__ = ["GLMData"]


def _is_numeric_dtype(array: Array) -> bool:
    return bool(jnp.issubdtype(array.dtype, jnp.number))


def _as_numeric_array(name: str, value: ArrayLike) -> Array:
    try:
        array = jnp.asarray(value)
    except TypeError as exc:
        raise TypeError(f"GLMData.{name} must be numeric.") from exc
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


class GLMData(eqx.Module, strict=True):
    """Canonical data noun for GLM workflows."""

    X: Array
    y: Array
    offset: Array | None = None
    weights: Array | None = None
    mask: Array | None = None

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> None:
        X = _as_numeric_array("X", X)
        y = _as_numeric_array("y", y)

        n_samples = X.shape[0]
        canonical_offset = None
        if offset is not None:
            canonical_offset = _canonicalize_numeric_vector("offset", offset, n_samples)

        canonical_weights = None
        if weights is not None:
            canonical_weights = _canonicalize_numeric_vector("weights", weights, n_samples)

        canonical_mask = None
        if mask is not None:
            canonical_mask = _canonicalize_mask(mask, n_samples)

        self.X = X
        self.y = y
        self.offset = canonical_offset
        self.weights = canonical_weights
        self.mask = canonical_mask

    def __check_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("GLMData.X must be rank-2 with shape (n, p).")
        if self.y.ndim != 1:
            raise ValueError("GLMData.y must be rank-1 with shape (n,).")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("GLMData.X and GLMData.y must share the sample dimension n.")

        _require_finite("X", self.X)
        _require_finite("y", self.y)

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
