# pattern: Functional Core

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

from jax import Array

from .data import GLMData


if TYPE_CHECKING:
    from .glm import GLM
    from .infer.diagnostics import Diagnostics


__all__ = ["Params", "FitResult", "Fitter", "validate_fit_result", "fit", "predict"]


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
        "se",
        "z",
        "p",
        "eta",
        "mu",
        "glm_wt",
        "diagnostics",
        "curvature",
        "score_residual",
    )
    return getattr(type(value), "__name__", None) == "FitResult" and all(
        hasattr(value, name) for name in required_fields
    )


class Params(NamedTuple):
    """Canonical model parameters."""

    beta: Array
    disp: Array


class FitResult(eqx.Module, strict=True):
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

    def __check_init__(self) -> None:
        if not _matches_namedtuple_shape(self.params, type_name="Params", fields=("beta", "disp")):
            raise TypeError("FitResult.params must be a Params instance.")
        if not _matches_namedtuple_shape(
            self.diagnostics,
            type_name="Diagnostics",
            fields=("converged", "num_iters", "objective", "objective_delta"),
        ):
            raise TypeError("FitResult.diagnostics must be a Diagnostics instance.")

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

        se = _as_contract_numeric_array("FitResult.se", self.se)
        if se.ndim != 1 or se.shape[0] != expected_p:
            raise ValueError("FitResult.se must be a rank-1 vector aligned with FitResult.params.beta.")
        _require_contract_finite("FitResult.se", se)

        z = _as_contract_numeric_array("FitResult.z", self.z)
        if z.ndim != 1 or z.shape[0] != expected_p:
            raise ValueError("FitResult.z must be a rank-1 vector aligned with FitResult.params.beta.")
        _require_contract_finite("FitResult.z", z)

        p = _as_contract_numeric_array("FitResult.p", self.p)
        if p.ndim != 1 or p.shape[0] != expected_p:
            raise ValueError("FitResult.p must be a rank-1 vector aligned with FitResult.params.beta.")
        _require_contract_finite("FitResult.p", p)

        curvature = _as_contract_numeric_array("FitResult.curvature", self.curvature)
        if curvature.ndim != 2 or curvature.shape[0] != curvature.shape[1]:
            raise ValueError("FitResult.curvature must be a square rank-2 matrix.")
        if curvature.shape[0] != expected_p:
            raise ValueError("FitResult.curvature shape must match FitResult.params.beta length.")
        _require_contract_finite("FitResult.curvature", curvature)

        eta = _as_contract_numeric_array("FitResult.eta", self.eta)
        if eta.ndim != 1:
            raise ValueError("FitResult.eta must be a rank-1 vector.")
        _require_contract_finite("FitResult.eta", eta)

        expected_n = eta.shape[0]

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

        converged = jnp.asarray(self.diagnostics.converged)
        if not jnp.issubdtype(converged.dtype, jnp.bool_):
            raise TypeError("FitResult.diagnostics.converged must be boolean.")
        if converged.ndim > 0 and converged.size != 1:
            raise ValueError("FitResult.diagnostics.converged must be scalar.")

        num_iters = _as_contract_numeric_array("FitResult.diagnostics.num_iters", self.diagnostics.num_iters)
        if num_iters.ndim > 0 and num_iters.size != 1:
            raise ValueError("FitResult.diagnostics.num_iters must be scalar.")
        _require_contract_finite("FitResult.diagnostics.num_iters", num_iters)

        objective = _as_contract_numeric_array("FitResult.diagnostics.objective", self.diagnostics.objective)
        if objective.ndim > 0 and objective.size != 1:
            raise ValueError("FitResult.diagnostics.objective must be scalar.")
        _require_contract_finite("FitResult.diagnostics.objective", objective)

        objective_delta = _as_contract_numeric_array(
            "FitResult.diagnostics.objective_delta",
            self.diagnostics.objective_delta,
        )
        if objective_delta.ndim > 0 and objective_delta.size != 1:
            raise ValueError("FitResult.diagnostics.objective_delta must be scalar.")
        _require_contract_finite("FitResult.diagnostics.objective_delta", objective_delta)


@runtime_checkable
class Fitter(Protocol):
    """Callable fit strategy over grammar nouns."""

    def __call__(self, model: GLM, data: GLMData, init: Params | None = None) -> FitResult:
        ...


def validate_fit_result(result: FitResult) -> None:
    """Validate FitResult artifacts used by infer/check verbs."""
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


class _ModelFitter:
    """Bridge canonical fit verb calls into the canonical GLM fit kernel."""

    def __call__(self, model: GLM, data: GLMData, init: Params | None = None) -> FitResult:
        from .glm import _fit_model

        X_array, _, offset_array, _, _ = data.canonical_arrays()
        init_beta, init_disp = _canonicalize_init(init, X_array.shape[1])
        init_eta = None if init_beta is None else X_array @ init_beta + offset_array
        return _fit_model(model, data, init_eta=init_eta, disp_init=init_disp)


DEFAULT_FITTER: Fitter = _ModelFitter()


def fit(model: GLM, data: GLMData, init: Params | None = None, *, fitter: Fitter = DEFAULT_FITTER) -> FitResult:
    """Canonical public fit verb over grammar nouns."""
    from .glm import GLM as _GLM

    if not isinstance(model, _GLM):
        raise TypeError("fit(...) expects `model` to be a GLM instance.")
    if not isinstance(data, GLMData):
        raise TypeError("fit(...) expects `data` to be a GLMData instance.")
    if init is not None and not _matches_namedtuple_shape(init, type_name="Params", fields=("beta", "disp")):
        raise TypeError("fit(...) expects `init` to be a Params instance or None.")
    if not callable(fitter):
        raise TypeError("fit(...) expects `fitter` to be callable.")
    result = fitter(model, data, init)
    if not _matches_fit_result_shape(result):
        raise TypeError("fit(...) expects `fitter` to return a FitResult instance.")
    validate_fit_result(result)
    return result


def predict(model: GLM, params: Params, data: GLMData) -> jnp.ndarray:
    """Pure prediction verb over grammar nouns."""
    from .glm import GLM as _GLM

    if not isinstance(model, _GLM):
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
