# pattern: Imperative Shell

from jaxtyping import ArrayLike, ScalarLike

from ..family.dist import ExponentialFamily
from .contracts import AbstractLinearSolver
from .fitters import (
    AbstractGLMFitter as AbstractGLMFitter,
    irls as _irls,
    IRLSFitter as IRLSFitter,
    IRLSState as IRLSState,
)


def irls(
    X: ArrayLike,
    y: ArrayLike,
    family: ExponentialFamily,
    solver: AbstractLinearSolver,
    eta: ArrayLike,
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
    offset_eta: ArrayLike = 0.0,
    alpha_init: ScalarLike = 0.0,
) -> IRLSState:
    r"""Compatibility shim for the historical IRLS function surface.

    **Arguments:**

    - `X`: Design matrix with shape `(n, p)`.
    - `y`: Response vector with shape `(n,)`.
    - `family`: Exponential-family distribution used for GLM updates.
    - `solver`: Linear-system strategy used for weighted least squares steps.
    - `eta`: Initial linear predictor.
    - `max_iter`: Maximum IRLS iterations.
    - `tol`: Convergence tolerance on likelihood deltas.
    - `step_size`: Step size for iterative updates.
    - `offset_eta`: Optional offset term in linear predictor space.
    - `alpha_init`: Initial dispersion parameter.

    **Returns:**

    - [`IRLSState`][glmax.infer.fitters.IRLSState] containing coefficient estimates,
      iteration count, convergence indicator, and final dispersion estimate.

    **Failure Modes:**

    - Boundary validation is enforced at `glmax.fit`/`GLM.fit` entrypoints.
      Invalid dtype, rank, shape, or finiteness constraints fail there with
      deterministic built-in exceptions before this compatibility shim is called.
    """
    return _irls(X, y, family, solver, eta, max_iter, tol, step_size, offset_eta, alpha_init)
