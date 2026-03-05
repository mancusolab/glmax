from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable, TYPE_CHECKING

from jax import Array
from jaxtyping import ArrayLike


if TYPE_CHECKING:
    from .glm import GLM


@dataclass(frozen=True)
class GLMData:
    """Canonical data noun for GLM workflows."""

    X: ArrayLike
    y: ArrayLike
    offset: ArrayLike | None = None
    weights: ArrayLike | None = None
    mask: ArrayLike | None = None


@dataclass(frozen=True)
class Params:
    """Canonical model parameters."""

    beta: Array
    disp: Array


@dataclass(frozen=True)
class Diagnostics:
    """Common diagnostics emitted by fit/check verbs."""

    converged: Array
    num_iters: Array


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
