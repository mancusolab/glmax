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


class GLMData(eqx.Module, strict=True):
    r"""Canonical data noun for GLM workflows.

    Wraps the design matrix, response vector, and optional nuisance arrays.
    All inputs are validated and canonicalized at construction time.
    """

    X: Array
    y: Array
    offset: Array | None = None
    weights: Array | None = None

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset: ArrayLike | None = None,
        weights: ArrayLike | None = None,
    ) -> None:
        r"""**Arguments:**

        - `X`: covariate matrix, shape `(n, p)`. Must be rank-2, finite, numeric.
        - `y`: response vector, shape `(n,)`. Must be rank-1, finite, numeric.
        - `offset`: optional additive offset in $\eta = X\beta + \text{offset}$,
          shape `(n,)` or broadcastable scalar.
        - `weights`: optional per-sample weights (reserved; not yet supported by `fit`).

        **Raises:**

        - `TypeError`: if `X` or `y` are non-numeric.
        - `ValueError`: if `X` is not rank-2, `y` is not rank-1, sample dimensions
          mismatch, or any finite-value check fails.
        """
        X = _as_numeric_array("X", X)
        y = _as_numeric_array("y", y)

        n_samples = X.shape[0]
        canonical_offset = None
        if offset is not None:
            canonical_offset = _canonicalize_numeric_vector("offset", offset, n_samples)

        canonical_weights = None
        if weights is not None:
            canonical_weights = _canonicalize_numeric_vector("weights", weights, n_samples)

        self.X = X
        self.y = y
        self.offset = canonical_offset
        self.weights = canonical_weights

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

    def canonical_arrays(self) -> tuple[Array, Array, Array, Array]:
        return (
            self.X,
            self.y,
            self.canonical_offset(),
            self.canonical_weights(),
        )
