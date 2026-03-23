# pattern: Functional Core

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx

from jax import Array
from jaxtyping import ArrayLike

from .._misc import inexact_asarray
from ..family import ExponentialDispersionFamily
from .irls import IRLSFitter
from .types import (
    AbstractFitter,
    FittedGLM,
    Params,
)


__all__ = ["fit", "predict"]


@eqx.filter_custom_jvp
def _fit_core(
    X: Array,
    y: Array,
    offset: Array,
    *,
    family: ExponentialDispersionFamily,
    init: Params | None,
    fitter: AbstractFitter,
) -> FittedGLM:
    result = fitter.fit(family, X, y, offset, None, init)
    return FittedGLM(family=family, result=result)


@_fit_core.def_jvp
def _fit_core_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array | None, Array | None, Array | None],
    *,
    family: ExponentialDispersionFamily,
    init: Params | None,
    fitter: AbstractFitter,
) -> tuple[FittedGLM, FittedGLM]:
    X, y, offset = primals
    dX, dy, doffset = tangents

    fitted = _fit_core(X, y, offset, family=family, init=init, fitter=fitter)
    beta = fitted.result.params.beta
    disp = fitted.result.params.disp
    aux = fitted.result.params.aux

    # Materialise symbolically-zero tangents.
    dX = jnp.zeros_like(X) if dX is None else dX
    dy = jnp.zeros_like(y) if dy is None else dy
    doffset = jnp.zeros_like(offset) if doffset is None else doffset

    # IFT for dbeta: at convergence ∇_β nll = 0.
    # Differentiating implicitly: H dbeta = -∂_data(∇_β nll) · d(data)
    # where H = X^T W X is the Fisher information evaluated at the converged fit.
    def score(X_, y_, offset_):
        return jax.grad(lambda b: family.negloglikelihood(y_, X_ @ b + offset_, disp, aux))(beta)

    _, rhs = jax.jvp(score, (X, y, offset), (dX, dy, doffset))
    # H = X^T W X is SPD under a well-specified GLM. Use lineax with throw=True so
    # that a singular H (e.g. rank-deficient design) raises rather than silently
    # producing NaN tangents — there is no status channel in a JVP path.
    H = X.T @ (fitted.glm_wt[:, None] * X)
    dbeta = lx.linear_solve(
        lx.MatrixLinearOperator(H, lx.positive_semidefinite_tag),
        -rhs,
        solver=lx.Cholesky(),
        throw=True,
    ).value

    # Linear predictor and mean tangents.
    deta = X @ dbeta + dX @ beta + doffset
    dmu = family.glink.inverse_deriv(fitted.eta) * deta

    # Nuisance parameter tangents via JVP through update_nuisance.
    def nuisance_fn(X_, y_, eta_):
        return family.update_nuisance(X_, y_, eta_, disp, step_size=1.0, aux=aux)

    _, (ddisp, daux) = jax.jvp(nuisance_fn, (X, y, fitted.eta), (dX, dy, deta))

    # GLM working weight tangent.
    def glm_wt_fn(eta_):
        _, _, w = family.calc_weight(eta_, disp, aux)
        return w

    _, dglm_wt = jax.jvp(glm_wt_fn, (fitted.eta,), (deta,))

    # Score residual tangent: d[(y - μ) g'(μ)].
    def score_res_fn(y_, mu_):
        return (y_ - mu_) * family.glink.deriv(mu_)

    _, dscore_res = jax.jvp(score_res_fn, (y, fitted.mu), (dy, dmu))

    # Objective tangent at converged beta.
    def objective_fn(X_, y_, offset_):
        return family.negloglikelihood(y_, X_ @ beta + offset_, disp, aux)

    _, dobjective = jax.jvp(objective_fn, (X, y, offset), (dX, dy, doffset))

    # Assemble the tangent FittedGLM via eqx.tree_at, which bypasses __check_init__.
    # Non-inexact leaves (converged: bool, num_iters: int) get float0 tangents, which
    # is the dtype JAX uses to represent tangents of non-differentiable values.
    _float0 = jax.dtypes.float0
    tangent = eqx.tree_at(
        lambda f: (
            f.result.params.beta,
            f.result.params.disp,
            f.result.X,
            f.result.y,
            f.result.eta,
            f.result.mu,
            f.result.glm_wt,
            f.result.converged,
            f.result.num_iters,
            f.result.objective,
            f.result.objective_delta,
            f.result.score_residual,
        ),
        fitted,
        (
            dbeta,
            ddisp,
            dX,
            dy,
            deta,
            dmu,
            dglm_wt,
            jnp.zeros((), dtype=_float0),
            jnp.zeros((), dtype=_float0),
            dobjective,
            jnp.zeros_like(fitted.objective_delta),
            dscore_res,
        ),
    )
    if aux is not None:
        tangent = eqx.tree_at(lambda f: f.result.params.aux, tangent, daux)

    return fitted, tangent


@eqx.filter_jit
def fit(
    family: ExponentialDispersionFamily,
    X: ArrayLike,
    y: ArrayLike,
    *,
    offset: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    init: Params | None = None,
    fitter: AbstractFitter = IRLSFitter(),
) -> FittedGLM:
    r"""Fit a GLM to observed data and return a fitted noun.

    This is the canonical `fit` grammar verb. It is `@eqx.filter_jit`-wrapped;
    the fitter strategy is treated as static structure under JIT. The returned
    value is a [`glmax.FittedGLM`][] noun that binds the family and the full
    [`glmax.FitResult`][] contract.

    Differentiation is supported via a registered `custom_jvp` rule based on
    the Implicit Function Theorem: at convergence the score is zero, so the
    tangent for $\hat\beta$ satisfies $H \, d\hat\beta = -\partial_{\text{data}}
    \nabla_\beta \ell \cdot d(\text{data})$, where $H = X^\top W X$ is the
    Fisher information at the converged fit.

    **Arguments:**

    - `family`: [`glmax.ExponentialDispersionFamily`][] instance.
    - `X`: covariate matrix, shape `(n, p)`.
    - `y`: response vector, shape `(n,)`.
    - `offset`: optional offset vector added to the linear predictor.
    - `weights`: optional per-sample weights (not yet supported).
    - `init`: optional [`glmax.Params`][] for warm-starting; `None` uses the
      family default.
    - `fitter`: [`glmax.AbstractFitter`][] strategy. Defaults to
      [`glmax.IRLSFitter`][].

    **Returns:**

    [`glmax.FittedGLM`][] noun binding the family and the
    [`glmax.FitResult`][] contract.

    **Raises:**

    - `TypeError`: if `family`, `init`, or `fitter` have wrong types.
    - `ValueError`: if `weights` is set (not yet supported).
    """

    if not isinstance(family, ExponentialDispersionFamily):
        raise TypeError("fit(...) expects `family` to be an ExponentialDispersionFamily instance.")
    if init is not None and not isinstance(init, Params):
        raise TypeError("fit(...) expects `init` to be a Params instance or None.")
    if not isinstance(fitter, AbstractFitter):
        raise TypeError("fit(...) expects `fitter` to be an AbstractFitter instance.")

    # ensure things are in inexact numerical space.
    X = cast(Array, inexact_asarray(X))
    y = cast(Array, inexact_asarray(y))
    if offset is None:
        offset = jnp.zeros_like(y)
    else:
        offset = cast(Array, inexact_asarray(offset))

    if weights is not None:
        raise ValueError("Per-sample weights are not supported yet.")

    if X.ndim != 2:
        raise ValueError("X must be rank-2 with shape (n, p).")
    if y.ndim != 1:
        raise ValueError("y must be rank-1 with shape (n,).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must share the sample dimension n.")

    # these are helpful enough, but lets not go overboard checking for bad input...
    # we need to re-cast due to error_if having type sig Any
    X = cast(Array, eqx.error_if(X, ~jnp.all(jnp.isfinite(X)), "X must contain only finite values."))
    y = cast(Array, eqx.error_if(y, ~jnp.all(jnp.isfinite(y)), "y must contain only finite values."))

    if init is not None:
        _, default_aux = family.init_nuisance()
        init = Params(beta=init.beta, disp=init.disp, aux=None if default_aux is None else init.aux)

    return _fit_core(X, y, offset, family=family, init=init, fitter=fitter)


@eqx.filter_jit
def predict(
    family: ExponentialDispersionFamily,
    params: Params,
    X: ArrayLike,
    *,
    offset: ArrayLike | None = None,
) -> Array:
    r"""Apply a fitted family to new data and return predicted means.

    This is the canonical `predict` grammar verb. It is `@eqx.filter_jit`-wrapped.
    Prediction computes $\hat{\mu} = g^{-1}(X \hat{\beta} + o)$, where $X$ is
    the design matrix, $\hat{\beta}$ is the fitted coefficient vector, $o$ is
    the optional offset, and $g$ is the link function.

    **Arguments:**

    - `family`: [`glmax.ExponentialDispersionFamily`][] instance.
    - `params`: fitted [`glmax.Params`][] (for example `fitted.params` from
      [`glmax.fit`][]).
    - `X`: covariate matrix, shape `(n, p)`.
    - `offset`: optional offset vector added to the linear predictor.

    **Returns:**

    Predicted mean response vector
    $\hat{\mu} = g^{-1}(X \hat{\beta} + o)$, shape `(n,)`.

    **Raises:**

    - `TypeError`: if `family`, `params`, or `X` have wrong types.
    """

    if not isinstance(family, ExponentialDispersionFamily):
        raise TypeError("predict(...) expects `family` to be an ExponentialDispersionFamily instance.")
    if not isinstance(params, Params):
        raise TypeError("predict(...) expects `params` to be a Params instance.")

    X = cast(Array, inexact_asarray(X))

    eta = X @ params.beta

    if offset is not None:
        offset = cast(Array, inexact_asarray(offset))
        eta = eta + offset

    return family.glink.inverse(eta)
