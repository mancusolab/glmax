# pattern: Functional Core

from abc import abstractmethod
from typing_extensions import TypeAlias

import equinox as eqx
import lineax as lx

from jaxtyping import Array, ArrayLike


SolverState: TypeAlias = tuple[lx.AbstractLinearOperator, ArrayLike]


class AbstractLinearSolver(eqx.Module, strict=True):
    """
    Define parent class for all solvers
    eta = X @ beta, the linear component
    """

    solver: eqx.AbstractVar[lx.AbstractLinearSolver]

    @abstractmethod
    def init(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> SolverState:
        pass

    def __call__(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> Array:
        """Linear equation solver

        **Arguments:**

        - :param X: covariate data matrix (nxp)
        - :param r: residuals
        - :param weights: weights for each individual

        **Returns:**
        The solution to the weighted least squares regression using QR factorization
        """
        A, b = self.init(X, r, weights)
        sol = lx.linear_solve(A, b, solver=self.solver)

        return sol.value
