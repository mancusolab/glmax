from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx

from jax import Array, numpy as jnp

from ..data import GLMData
from ..glm import GLM
from .solve import AbstractLinearSolver


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
        if not isinstance(self.params, Params):
            raise TypeError("FitResult.params must be a Params instance.")

        beta = jnp.asarray(self.params.beta)
        if not jnp.issubdtype(beta.dtype, jnp.inexact):
            raise TypeError("FitResult.params.beta must have an inexact dtype.")
        if beta.ndim != 1:
            raise ValueError("FitResult.params.beta must be a rank-1 vector.")

        disp = jnp.asarray(self.params.disp)
        if not jnp.issubdtype(disp.dtype, jnp.inexact):
            raise TypeError("FitResult.params.disp must have an inexact dtype.")
        if disp.ndim > 0 and disp.size != 1:
            raise ValueError("FitResult.params.disp must be a scalar.")

        expected_p = beta.shape[0]

        X = jnp.asarray(self.X)
        if X.ndim != 2:
            raise ValueError("FitResult.X must be a rank-2 matrix.")
        if X.shape[1] != expected_p:
            raise ValueError("FitResult.X shape must match FitResult.params.beta length.")

        y = jnp.asarray(self.y)
        if y.ndim != 1:
            raise ValueError("FitResult.y must be a rank-1 vector.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("FitResult.y must align with FitResult.X over samples.")

        expected_n = X.shape[0]

        eta = jnp.asarray(self.eta)
        if eta.ndim != 1 or eta.shape[0] != expected_n:
            raise ValueError("FitResult.eta must be a rank-1 vector aligned with FitResult.X over samples.")

        mu = jnp.asarray(self.mu)
        if mu.ndim != 1 or mu.shape[0] != expected_n:
            raise ValueError("FitResult.mu must be a rank-1 vector aligned with FitResult.eta.")

        glm_wt = jnp.asarray(self.glm_wt)
        if glm_wt.ndim != 1 or glm_wt.shape[0] != expected_n:
            raise ValueError("FitResult.glm_wt must be a rank-1 vector aligned with FitResult.eta.")

        score_residual = jnp.asarray(self.score_residual)
        if score_residual.ndim != 1 or score_residual.shape[0] != expected_n:
            raise ValueError("FitResult.score_residual must be a rank-1 vector aligned with FitResult.eta.")

        converged = jnp.asarray(self.converged)
        if not jnp.issubdtype(converged.dtype, jnp.bool_):
            raise TypeError("FitResult.converged must be boolean.")
        if converged.ndim > 0 and converged.size != 1:
            raise ValueError("FitResult.converged must be scalar.")

        num_iters = jnp.asarray(self.num_iters)
        if num_iters.ndim > 0 and num_iters.size != 1:
            raise ValueError("FitResult.num_iters must be scalar.")

        objective = jnp.asarray(self.objective)
        if objective.ndim > 0 and objective.size != 1:
            raise ValueError("FitResult.objective must be scalar.")

        objective_delta = jnp.asarray(self.objective_delta)
        if objective_delta.ndim > 0 and objective_delta.size != 1:
            raise ValueError("FitResult.objective_delta must be scalar.")


class FittedGLM(eqx.Module, strict=True):
    """Canonical fitted-model noun used by downstream grammar verbs."""

    model: GLM
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
        if not isinstance(self.model, GLM):
            raise TypeError("FittedGLM.model must be a GLM instance.")
        if not isinstance(self.result, FitResult):
            raise TypeError("FittedGLM.result must be a FitResult instance.")


class AbstractFitter(eqx.Module, strict=True):
    """Abstract fit strategy over grammar nouns."""

    solver: eqx.AbstractVar[AbstractLinearSolver]

    @abstractmethod
    def __call__(self, model: GLM, data: GLMData, init: Params | None = None) -> FitResult:
        """Fit `model` against `data` with optional parameter initialisation."""


def _canonicalize_init(init: Params | None, n_features: int) -> tuple[Array | None, Array | None]:
    if init is None:
        return None, None

    try:
        beta = jnp.asarray(init.beta)
    except TypeError as exc:
        raise TypeError("Params.beta must be numeric.") from exc
    if not jnp.issubdtype(beta.dtype, jnp.inexact):
        raise TypeError("Params.beta must have an inexact dtype.")
    if beta.ndim != 1 or beta.shape[0] != n_features:
        raise ValueError("Params.beta must be a one-dimensional vector with length equal to X.shape[1].")

    try:
        disp = jnp.asarray(init.disp)
    except TypeError as exc:
        raise TypeError("Params.disp must be numeric.") from exc
    if not jnp.issubdtype(disp.dtype, jnp.inexact):
        raise TypeError("Params.disp must have an inexact dtype.")
    if disp.ndim > 0 and disp.size != 1:
        raise ValueError("Params.disp must be a scalar.")

    return beta, disp
