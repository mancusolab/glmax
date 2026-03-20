# pattern: Functional Core


import equinox as eqx

from jax import Array, numpy as jnp
from jaxtyping import ArrayLike

from ..data import GLMData
from ..family import ExponentialDispersionFamily
from .irls import IRLSFitter
from .types import (
    _canonicalize_init,
    AbstractFitter,
    FitResult,
    FittedGLM,
    Params,
)


__all__ = ["fit", "predict"]


def _validated_params(params: Params, n_features: int) -> Params:
    beta, disp, aux = _canonicalize_init(params, n_features)
    assert beta is not None and disp is not None
    return Params(beta=beta, disp=disp, aux=aux)


def _normalize_init_aux(family: ExponentialDispersionFamily, params: Params) -> Params:
    _, default_aux = family.init_nuisance()
    if default_aux is None and params.aux is not None:
        return Params(beta=params.beta, disp=params.disp, aux=None)
    return params


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

    - `TypeError`: if `family`, `init`, or `fitter` have wrong types,
      or if the fitter does not return a `FitResult`.
    - `ValueError`: if `weights` is set (not yet supported).
    """

    if not isinstance(family, ExponentialDispersionFamily):
        raise TypeError("fit(...) expects `family` to be an ExponentialDispersionFamily instance.")
    if init is not None and not isinstance(init, Params):
        raise TypeError("fit(...) expects `init` to be a Params instance or None.")
    if not isinstance(fitter, AbstractFitter):
        raise TypeError("fit(...) expects `fitter` to be an AbstractFitter instance.")

    data = GLMData(X=X, y=y, offset=offset, weights=weights)

    if init is not None:
        init = _validated_params(init, data.X.shape[1])
        if not isinstance(fitter, IRLSFitter):
            init = _normalize_init_aux(family, init)

    result = fitter(family, data, init)

    if not isinstance(result, FitResult):
        raise TypeError("fit(...) expects `fitter` to return a FitResult instance.")

    return FittedGLM(family=family, result=result)


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

    X_array = jnp.asarray(X)
    offset_array = jnp.zeros(X_array.shape[0]) if offset is None else jnp.asarray(offset)
    params = _normalize_init_aux(family, _validated_params(params, X_array.shape[1]))

    eta = X_array @ params.beta + offset_array
    return family.glink.inverse(eta)
