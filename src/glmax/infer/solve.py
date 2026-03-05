"""Internal linear solver adapters used by GLM fit/infer kernels."""

from abc import abstractmethod
from typing_extensions import TypeAlias

import equinox as eqx
import jax.numpy as jnp
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


class QRSolver(AbstractLinearSolver, strict=True):
    """
    Define parent class for all solvers
    If the operator is non-square
    If your primary concern is computational efficiency
    Note that whilst this does handle non-square operators, it still can only handle full-rank operators.
    """

    solver: lx.AbstractLinearSolver = lx.QR()

    def init(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> SolverState:
        """
        **Arguments:**

         - :param X: covariate data matrix (nxp)
         - :param r: residuals
         - :param weights: weights for each individual

            **Returns:**
            The solution to the weighted least squares regression using QR factorization.
        """
        w_half = jnp.sqrt(weights)
        w_half_r = w_half * r
        w_half_X = X * w_half[:, jnp.newaxis]

        A = lx.MatrixLinearOperator(w_half_X)

        return A, w_half_r


class CholeskySolver(AbstractLinearSolver):
    """
    Define parent class for all solvers
    Preferred solver for positive or negative definite systems
    The operator must be square, nonsingular, and either positive or negative definite.
    """

    solver: lx.AbstractLinearSolver = lx.Cholesky()

    def init(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> SolverState:
        Xw = X * weights[:, jnp.newaxis]
        A = lx.MatrixLinearOperator(Xw.T @ X, tags=lx.positive_semidefinite_tag)
        b = Xw.T @ r

        return A, b


class CGSolver(AbstractLinearSolver):
    """
    **Arguments:**

           - :param X: covariate data matrix (nxp)
           - :param r: residuals
           - :param weights: weights for each individual

              **Returns:**
              The solution to the weighted least squares regression using
              Cholesky decomposition on the coefficient matrix.
    """

    solver: lx.AbstractLinearSolver = lx.NormalCG(atol=1e-5, rtol=1e-5)

    def init(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> SolverState:
        w_half = jnp.sqrt(weights)
        w_half_X = X * w_half[:, jnp.newaxis]

        # Here we solve (XtWX) beta = XtW b, so A = X * sqrt(W), b = sqrt(W) * r
        A = lx.MatrixLinearOperator(w_half_X)
        b = w_half * r

        return A, b
