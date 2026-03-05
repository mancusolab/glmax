# pattern: Functional Core

from abc import abstractmethod
from typing_extensions import TypeAlias

import equinox as eqx
import lineax as lx

from jaxtyping import Array, ArrayLike


SolverState: TypeAlias = tuple[lx.AbstractLinearOperator, ArrayLike]


class AbstractLinearSolver(eqx.Module, strict=True):
    r"""Abstract contract for weighted least-squares linear solvers.

    **Arguments:**

    - Implementations define a concrete `solver` strategy and `init(...)` mapping
      from weighted regression inputs to a linear operator/right-hand side pair.

    **Returns:**

    - Concrete implementations return coefficient estimates compatible with
      GLM fitter update loops.

    **Failure Modes:**

    - Implementations may raise backend-specific linear-algebra errors if the
      weighted system is ill-posed for the configured solver strategy.
    """

    solver: eqx.AbstractVar[lx.AbstractLinearSolver]

    @abstractmethod
    def init(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> SolverState:
        pass

    def __call__(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> Array:
        r"""Solve one weighted least-squares linear system.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `r`: Working-response vector with shape `(n,)`.
        - `weights`: Per-sample non-negative weights with shape `(n,)`.

        **Returns:**

        - Coefficient update vector with shape `(p,)`.

        **Failure Modes:**

        - Raises backend-specific linear solve errors if the configured solver
          cannot handle the induced operator properties.
        """
        A, b = self.init(X, r, weights)
        sol = lx.linear_solve(A, b, solver=self.solver)

        return sol.value
