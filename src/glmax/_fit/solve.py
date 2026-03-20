# pattern: Functional Core

"""Internal linear solver adapters used by GLM fit/_infer kernels."""

from abc import abstractmethod
from typing_extensions import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import lineax as lx

from jax import Array


# from jaxtyping import Array, ArrayLike


SolverState: TypeAlias = tuple[lx.AbstractLinearOperator, Array]


class AbstractLinearSolver(eqx.Module, strict=True):
    r"""Abstract contract for weighted least-squares linear solvers.

    Subclasses must provide a concrete `solver: lx.AbstractLinearSolver` field
    and implement `init(...)` to map weighted regression inputs to a linear
    operator / right-hand-side pair. Concrete implementations may raise
    backend-specific linear-algebra errors for ill-posed systems.
    """

    solver: eqx.AbstractVar[lx.AbstractLinearSolver]

    @abstractmethod
    def init(self, X: Array, r: Array, weights: Array) -> SolverState:
        pass

    def __call__(self, X: Array, r: Array, weights: Array) -> Array:
        r"""Solve one weighted least-squares linear system.

        This solves the weighted least-squares problem induced by design
        matrix $X$, working response $r$, and non-negative weights $w$.

        **Arguments:**

        - `X`: covariate matrix $X$, shape `(n, p)`.
        - `r`: working-response vector $r$, shape `(n,)`.
        - `weights`: per-sample non-negative weight vector $w$, shape `(n,)`.

        **Returns:**

        - Coefficient update vector with shape `(p,)`.

        **Failure Modes:**

        - Raises backend-specific linear solve errors if the configured solver
          cannot handle the induced operator properties.
        """
        A, b = self.init(X, r, weights)
        sol = lx.linear_solve(A, b, solver=self.solver)

        return sol.value


class QRSolver(AbstractLinearSolver, strict=True):
    r"""QR-based weighted least-squares solver.

    Uses `lineax.QR` to solve full-rank weighted systems; supports non-square
    operators. May fail for rank-deficient systems.
    """

    solver: lx.AbstractLinearSolver

    def __init__(self, solver: lx.AbstractLinearSolver = lx.QR()) -> None:
        r"""Construct a QR-based solver.

        **Arguments:**

        - `solver`: `lineax` solver instance (default: `lx.QR()`).
        """
        self.solver = solver

    def init(self, X: Array, r: Array, weights: Array) -> SolverState:
        r"""Build weighted QR operator inputs.

        This uses $\sqrt{w}$ so the least-squares objective is
        $\lVert \sqrt{W}(X \beta - r)\rVert_2^2$, where
        $W = \operatorname{diag}(w)$.

        **Arguments:**

        - `X`: covariate matrix $X$, shape `(n, p)`.
        - `r`: working-response vector $r$, shape `(n,)`.
        - `weights`: per-sample weight vector $w$, shape `(n,)`.

        **Returns:**

        - Tuple `(A, b)` containing a linear operator and right-hand side.

        **Failure Modes:**

        - Invalid weight broadcasting or incompatible input shapes propagate as
          backend array-shape errors.
        """
        w_half = jnp.sqrt(weights)
        w_half_r = w_half * r
        w_half_X = X * w_half[:, jnp.newaxis]

        A = lx.MatrixLinearOperator(w_half_X)

        return A, w_half_r


class CholeskySolver(AbstractLinearSolver):
    r"""Cholesky-based weighted least-squares solver.

    Uses `lineax.Cholesky` on normal-equation operators. Requires numerically
    stable normal equations; near-singular systems may fail.
    """

    solver: lx.AbstractLinearSolver

    def __init__(self, solver: lx.AbstractLinearSolver = lx.Cholesky()) -> None:
        r"""Construct a Cholesky-based solver.

        **Arguments:**

        - `solver`: `lineax` solver instance (default: `lx.Cholesky()`).
        """
        self.solver = solver

    def init(self, X: Array, r: Array, weights: Array) -> SolverState:
        Xw = X * weights[:, jnp.newaxis]
        A = lx.MatrixLinearOperator(Xw.T @ X, tags=lx.positive_semidefinite_tag)
        b = Xw.T @ r

        return A, b


class CGSolver(AbstractLinearSolver):
    r"""Conjugate-gradient weighted least-squares solver.

    Uses `lineax.NormalCG` for iterative weighted least-squares solves.
    Convergence may degrade on poorly conditioned systems.
    """

    solver: lx.AbstractLinearSolver

    def __init__(self, solver: lx.AbstractLinearSolver = lx.Normal(lx.CG(atol=1e-5, rtol=1e-5))) -> None:
        r"""Construct a conjugate-gradient solver.

        **Arguments:**

        - `solver`: `lineax` solver instance (default: `lx.Normal(lx.CG(atol=1e-5, rtol=1e-5))`).
        """
        self.solver = solver

    def init(self, X: Array, r: Array, weights: Array) -> SolverState:
        w_half = jnp.sqrt(weights)
        w_half_X = X * w_half[:, jnp.newaxis]

        # Here we solve (XtWX) beta = XtW b, so A = X * sqrt(W), b = sqrt(W) * r
        A = lx.MatrixLinearOperator(w_half_X)
        b = w_half * r

        return A, b
