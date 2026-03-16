from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple, TYPE_CHECKING

import equinox as eqx

from jax import Array, numpy as jnp

from ..data import GLMData


if TYPE_CHECKING:
    from ..glm import GLM


class Params(NamedTuple):
    """Canonical model parameters."""

    beta: Array
    disp: Array


class FitResult(eqx.Module, strict=True):
    """Canonical fit contract shared by grammar verbs."""

    params: Params
    X: Array
    y: Array
    eta: Array
    mu: Array
    glm_wt: Array
    converged: Array
    num_iters: Array
    objective: Array
    objective_delta: Array
    score_residual: Array

    @property
    def beta(self) -> Array:
        return self.params.beta

    def __check_init__(self) -> None:
        if not _matches_namedtuple_shape(self.params, type_name="Params", fields=("beta", "disp")):
            raise TypeError("FitResult.params must be a Params instance.")

        beta = _as_contract_numeric_array("FitResult.params.beta", self.params.beta)
        if beta.ndim != 1:
            raise ValueError("FitResult.params.beta must be a rank-1 vector.")
        _require_contract_inexact_dtype("FitResult.params.beta", beta)
        _require_contract_finite("FitResult.params.beta", beta)

        disp = _as_contract_numeric_array("FitResult.params.disp", self.params.disp)
        if disp.ndim > 0 and disp.size != 1:
            raise ValueError("FitResult.params.disp must be a scalar.")
        _require_contract_inexact_dtype("FitResult.params.disp", disp)
        _require_contract_finite("FitResult.params.disp", disp)

        expected_p = beta.shape[0]

        X = _as_contract_numeric_array("FitResult.X", self.X)
        if X.ndim != 2:
            raise ValueError("FitResult.X must be a rank-2 matrix.")
        _require_contract_finite("FitResult.X", X)
        if X.shape[1] != expected_p:
            raise ValueError("FitResult.X shape must match FitResult.params.beta length.")

        y = _as_contract_numeric_array("FitResult.y", self.y)
        if y.ndim != 1:
            raise ValueError("FitResult.y must be a rank-1 vector.")
        _require_contract_finite("FitResult.y", y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("FitResult.y must align with FitResult.X over samples.")

        eta = _as_contract_numeric_array("FitResult.eta", self.eta)
        if eta.ndim != 1:
            raise ValueError("FitResult.eta must be a rank-1 vector.")
        _require_contract_finite("FitResult.eta", eta)

        expected_n = X.shape[0]
        if eta.shape[0] != expected_n:
            raise ValueError("FitResult.eta must align with FitResult.X over samples.")

        mu = _as_contract_numeric_array("FitResult.mu", self.mu)
        if mu.ndim != 1 or mu.shape[0] != expected_n:
            raise ValueError("FitResult.mu must be a rank-1 vector aligned with FitResult.eta.")
        _require_contract_finite("FitResult.mu", mu)

        glm_wt = _as_contract_numeric_array("FitResult.glm_wt", self.glm_wt)
        if glm_wt.ndim != 1 or glm_wt.shape[0] != expected_n:
            raise ValueError("FitResult.glm_wt must be a rank-1 vector aligned with FitResult.eta.")
        _require_contract_finite("FitResult.glm_wt", glm_wt)

        score_residual = _as_contract_numeric_array("FitResult.score_residual", self.score_residual)
        if score_residual.ndim != 1 or score_residual.shape[0] != expected_n:
            raise ValueError("FitResult.score_residual must be a rank-1 vector aligned with FitResult.eta.")
        _require_contract_finite("FitResult.score_residual", score_residual)

        converged = jnp.asarray(self.converged)
        if not jnp.issubdtype(converged.dtype, jnp.bool_):
            raise TypeError("FitResult.converged must be boolean.")
        if converged.ndim > 0 and converged.size != 1:
            raise ValueError("FitResult.converged must be scalar.")

        num_iters = _as_contract_numeric_array("FitResult.num_iters", self.num_iters)
        if num_iters.ndim > 0 and num_iters.size != 1:
            raise ValueError("FitResult.num_iters must be scalar.")
        _require_contract_finite("FitResult.num_iters", num_iters)

        objective = _as_contract_numeric_array("FitResult.objective", self.objective)
        if objective.ndim > 0 and objective.size != 1:
            raise ValueError("FitResult.objective must be scalar.")
        _require_contract_finite("FitResult.objective", objective)

        objective_delta = _as_contract_numeric_array("FitResult.objective_delta", self.objective_delta)
        if objective_delta.ndim > 0 and objective_delta.size != 1:
            raise ValueError("FitResult.objective_delta must be scalar.")
        _require_contract_finite("FitResult.objective_delta", objective_delta)


class FittedGLM(eqx.Module, strict=True):
    """Canonical fitted-model noun used by downstream grammar verbs."""

    model: "GLM"
    result: FitResult

    @property
    def params(self) -> Params:
        return self.result.params

    @property
    def X(self) -> Array:
        return self.result.X

    @property
    def y(self) -> Array:
        return self.result.y

    @property
    def beta(self) -> Array:
        return self.result.beta

    @property
    def eta(self) -> Array:
        return self.result.eta

    @property
    def mu(self) -> Array:
        return self.result.mu

    @property
    def glm_wt(self) -> Array:
        return self.result.glm_wt

    @property
    def converged(self) -> Array:
        return self.result.converged

    @property
    def num_iters(self) -> Array:
        return self.result.num_iters

    @property
    def objective(self) -> Array:
        return self.result.objective

    @property
    def objective_delta(self) -> Array:
        return self.result.objective_delta

    @property
    def score_residual(self) -> Array:
        return self.result.score_residual

    def __check_init__(self) -> None:
        from ..glm import GLM as _GLM

        if not isinstance(self.model, _GLM):
            raise TypeError("FittedGLM.model must be a GLM instance.")
        if not _matches_fit_result_shape(self.result):
            raise TypeError("FittedGLM.result must be a FitResult instance.")
        validate_fit_result(self.result)


class Fitter(eqx.Module, strict=True):
    """Abstract fit strategy over grammar nouns."""

    @abstractmethod
    def __call__(self, model: GLM, data: GLMData, init: Params | None = None) -> FitResult:
        """Fit `model` against `data` with optional parameter initialisation."""


def _is_numeric_dtype(array: Array) -> bool:
    return bool(jnp.issubdtype(array.dtype, jnp.number))


def _as_contract_numeric_array(name: str, value: object) -> Array:
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


def _matches_namedtuple_shape(value: object, *, type_name: str, fields: tuple[str, ...]) -> bool:
    return (
        isinstance(value, tuple)
        and getattr(type(value), "__name__", None) == type_name
        and tuple(getattr(value, "_fields", ())) == fields
    )


def _matches_fit_result_shape(value: object) -> bool:
    required_fields = (
        "params",
        "X",
        "y",
        "eta",
        "mu",
        "glm_wt",
        "converged",
        "num_iters",
        "objective",
        "objective_delta",
        "score_residual",
    )
    return getattr(type(value), "__name__", None) == "FitResult" and all(
        hasattr(value, name) for name in required_fields
    )


def _matches_fitted_glm_shape(value: object) -> bool:
    required_fields = ("model", "result")
    return getattr(type(value), "__name__", None) == "FittedGLM" and all(
        hasattr(value, name) for name in required_fields
    )


def _matches_fitter_shape(value: object) -> bool:
    valid_modules = {"glmax.fit", "glmax._fit.fit", "glmax._fit.types"}
    return any(base.__name__ == "Fitter" and base.__module__ in valid_modules for base in type(value).__mro__)


def validate_fit_result(result: FitResult) -> None:
    """Validate FitResult artifacts used by _infer/check verbs."""
    if not _matches_fit_result_shape(result):
        raise TypeError("validate_fit_result(...) expects `result` to be a FitResult instance.")
    result.__check_init__()


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
