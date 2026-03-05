# pattern: Functional Core

import jax.numpy as jnp
import lineax as lx

from jaxtyping import ArrayLike

from .contracts import AbstractLinearSolver, SolverState


class QRSolver(AbstractLinearSolver, strict=True):
    r"""QR-based weighted least-squares solver.

    **Arguments:**

    - Uses `lineax.QR` to solve full-rank weighted systems and supports
      non-square operators.

    **Returns:**

    - Weighted least-squares coefficient updates for each IRLS iteration.

    **Failure Modes:**

    - May fail for rank-deficient systems where QR assumptions are violated.
    """

    solver: lx.AbstractLinearSolver = lx.QR()

    def init(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> SolverState:
        r"""Build weighted QR operator inputs.

        **Arguments:**

        - `X`: Covariate matrix with shape `(n, p)`.
        - `r`: Working-response vector with shape `(n,)`.
        - `weights`: Per-sample weights with shape `(n,)`.

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

    **Arguments:**

    - Uses `lineax.Cholesky` on normal-equation operators.

    **Returns:**

    - Weighted least-squares coefficient updates for each IRLS iteration.

    **Failure Modes:**

    - Requires numerically stable normal equations; near-singular systems may fail.
    """

    solver: lx.AbstractLinearSolver = lx.Cholesky()

    def init(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> SolverState:
        Xw = X * weights[:, jnp.newaxis]
        A = lx.MatrixLinearOperator(Xw.T @ X, tags=lx.positive_semidefinite_tag)
        b = Xw.T @ r

        return A, b


class CGSolver(AbstractLinearSolver):
    r"""Conjugate-gradient weighted least-squares solver.

    **Arguments:**

    - Uses `lineax.NormalCG` for iterative weighted least-squares solves.

    **Returns:**

    - Weighted least-squares coefficient updates for each IRLS iteration.

    **Failure Modes:**

    - Convergence may degrade on poorly conditioned systems depending on tolerance settings.
    """

    solver: lx.AbstractLinearSolver = lx.NormalCG(atol=1e-5, rtol=1e-5)

    def init(self, X: ArrayLike, r: ArrayLike, weights: ArrayLike) -> SolverState:
        w_half = jnp.sqrt(weights)
        w_half_X = X * w_half[:, jnp.newaxis]

        # Here we solve (XtWX) beta = XtW b, so A = X * sqrt(W), b = sqrt(W) * r
        A = lx.MatrixLinearOperator(w_half_X)
        b = w_half * r

        return A, b
