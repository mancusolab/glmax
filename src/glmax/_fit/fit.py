# pattern: Functional Core

from typing import cast

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jaxtyping import ArrayLike

from .._misc import inexact_asarray
from ..family import ExponentialDispersionFamily
from .irls import IRLSFitter
from .types import (
    AbstractFitter,
    FitResult,
    FittedGLM,
    Params,
)


__all__ = ["fit", "predict"]


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

    # ensure things are in inexact numerical space.
    X = cast(Array, inexact_asarray(X))
    y = cast(Array, inexact_asarray(y))
    if offset is None:
        offset = jnp.zeros_like(y)
    else:
        offset = cast(Array, inexact_asarray(offset))

    if weights is not None:
        weights = cast(Array, inexact_asarray(weights))

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

    result = fitter.fit(family, X, y, offset, weights, init)

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

    X = cast(Array, inexact_asarray(X))

    eta = X @ params.beta

    if offset is not None:
        offset = cast(Array, inexact_asarray(offset))
        eta = eta + offset

    return family.glink.inverse(eta)
