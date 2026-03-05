# pattern: Functional Core

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


def _as_contract_numeric_array(name: str, value: ArrayLike) -> Array:
    try:
        array = jnp.asarray(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be numeric.") from exc
    if not _is_numeric_dtype(array):
        raise TypeError(f"{name} must be numeric.")
    return array


def _require_contract_finite(name: str, array: Array) -> None:
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError(f"{name} must contain only finite values.")


def _require_contract_inexact_dtype(name: str, array: Array) -> None:
    if not bool(jnp.issubdtype(array.dtype, jnp.inexact)):
        raise TypeError(f"{name} must have an inexact dtype.")


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
    objective: Array
    objective_delta: Array


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
    curvature: Array
    score_residual: Array

    @property
    def beta(self) -> Array:
        return self.params.beta

    @property
    def num_iters(self) -> Array:
        return self.diagnostics.num_iters

    @property
    def converged(self) -> Array:
        return self.diagnostics.converged

    @property
    def objective(self) -> Array:
        return self.diagnostics.objective

    @property
    def objective_delta(self) -> Array:
        return self.diagnostics.objective_delta


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


def validate_fit_result(result: FitResult) -> None:
    """Validate FitResult artifacts used by infer/check verbs."""
    if not isinstance(result.params, Params):
        raise TypeError("FitResult.params must be a Params instance.")
    if not isinstance(result.diagnostics, Diagnostics):
        raise TypeError("FitResult.diagnostics must be a Diagnostics instance.")

    beta = _as_contract_numeric_array("FitResult.params.beta", result.params.beta)
    if beta.ndim != 1:
        raise ValueError("FitResult.params.beta must be a rank-1 vector.")
    _require_contract_inexact_dtype("FitResult.params.beta", beta)
    _require_contract_finite("FitResult.params.beta", beta)

    disp = _as_contract_numeric_array("FitResult.params.disp", result.params.disp)
    if disp.ndim > 0 and disp.size != 1:
        raise ValueError("FitResult.params.disp must be a scalar.")
    _require_contract_inexact_dtype("FitResult.params.disp", disp)
    _require_contract_finite("FitResult.params.disp", disp)

    expected_p = beta.shape[0]

    se = _as_contract_numeric_array("FitResult.se", result.se)
    if se.ndim != 1 or se.shape[0] != expected_p:
        raise ValueError("FitResult.se must be a rank-1 vector aligned with FitResult.params.beta.")
    _require_contract_finite("FitResult.se", se)

    z = _as_contract_numeric_array("FitResult.z", result.z)
    if z.ndim != 1 or z.shape[0] != expected_p:
        raise ValueError("FitResult.z must be a rank-1 vector aligned with FitResult.params.beta.")
    _require_contract_finite("FitResult.z", z)

    p = _as_contract_numeric_array("FitResult.p", result.p)
    if p.ndim != 1 or p.shape[0] != expected_p:
        raise ValueError("FitResult.p must be a rank-1 vector aligned with FitResult.params.beta.")
    _require_contract_finite("FitResult.p", p)

    curvature = _as_contract_numeric_array("FitResult.curvature", result.curvature)
    if curvature.ndim != 2 or curvature.shape[0] != curvature.shape[1]:
        raise ValueError("FitResult.curvature must be a square rank-2 matrix.")
    if curvature.shape[0] != expected_p:
        raise ValueError("FitResult.curvature shape must match FitResult.params.beta length.")
    _require_contract_finite("FitResult.curvature", curvature)

    eta = _as_contract_numeric_array("FitResult.eta", result.eta)
    if eta.ndim != 1:
        raise ValueError("FitResult.eta must be a rank-1 vector.")
    _require_contract_finite("FitResult.eta", eta)

    expected_n = eta.shape[0]

    mu = _as_contract_numeric_array("FitResult.mu", result.mu)
    if mu.ndim != 1 or mu.shape[0] != expected_n:
        raise ValueError("FitResult.mu must be a rank-1 vector aligned with FitResult.eta.")
    _require_contract_finite("FitResult.mu", mu)

    glm_wt = _as_contract_numeric_array("FitResult.glm_wt", result.glm_wt)
    if glm_wt.ndim != 1 or glm_wt.shape[0] != expected_n:
        raise ValueError("FitResult.glm_wt must be a rank-1 vector aligned with FitResult.eta.")
    _require_contract_finite("FitResult.glm_wt", glm_wt)

    score_residual = _as_contract_numeric_array("FitResult.score_residual", result.score_residual)
    if score_residual.ndim != 1 or score_residual.shape[0] != expected_n:
        raise ValueError("FitResult.score_residual must be a rank-1 vector aligned with FitResult.eta.")
    _require_contract_finite("FitResult.score_residual", score_residual)

    converged = jnp.asarray(result.diagnostics.converged)
    if not jnp.issubdtype(converged.dtype, jnp.bool_):
        raise TypeError("FitResult.diagnostics.converged must be boolean.")
    if converged.ndim > 0 and converged.size != 1:
        raise ValueError("FitResult.diagnostics.converged must be scalar.")

    num_iters = _as_contract_numeric_array("FitResult.diagnostics.num_iters", result.diagnostics.num_iters)
    if num_iters.ndim > 0 and num_iters.size != 1:
        raise ValueError("FitResult.diagnostics.num_iters must be scalar.")
    _require_contract_finite("FitResult.diagnostics.num_iters", num_iters)

    objective = _as_contract_numeric_array("FitResult.diagnostics.objective", result.diagnostics.objective)
    if objective.ndim > 0 and objective.size != 1:
        raise ValueError("FitResult.diagnostics.objective must be scalar.")
    _require_contract_finite("FitResult.diagnostics.objective", objective)

    objective_delta = _as_contract_numeric_array(
        "FitResult.diagnostics.objective_delta",
        result.diagnostics.objective_delta,
    )
    if objective_delta.ndim > 0 and objective_delta.size != 1:
        raise ValueError("FitResult.diagnostics.objective_delta must be scalar.")
    _require_contract_finite("FitResult.diagnostics.objective_delta", objective_delta)
