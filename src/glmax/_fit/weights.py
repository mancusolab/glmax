# pattern: Functional Core

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from jax import Array
from jaxtyping import ArrayLike

from .._misc import inexact_asarray


__all__ = ["weights"]


def _check_or_error_if(value: Array, pred: Array, message: str) -> Array:
    try:
        if bool(pred):
            raise ValueError(message)
        return value
    except TypeError:
        return eqx.error_if(value, pred, message)


class AbstractWeights(eqx.Module, strict=True):
    r"""Abstract base for semantic sample-weight specifications."""

    value: eqx.AbstractVar[Array]

    @abstractmethod
    def fit_multiplier(self) -> Array:
        r"""Return the multiplier used in fitting equations."""

    @abstractmethod
    def objective_multiplier(self) -> Array:
        r"""Return the multiplier used for likelihood/objective sums."""

    @abstractmethod
    def effective_n(self, n_rows: int) -> Array:
        r"""Return the effective sample size for residual degrees of freedom."""


class FrequencyWeights(AbstractWeights, strict=True):
    r"""Frequency weights: row `i` represents `value[i]` repeated observations."""

    value: Array

    def __init__(self, value: ArrayLike) -> None:
        value = inexact_asarray(value)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError("frequency weights must have an inexact dtype.")
        if value.ndim != 1:
            raise ValueError("frequency weights must be rank-1 with shape (n,).")
        value = _check_or_error_if(
            value,
            ~jnp.all(jnp.isfinite(value)),
            "frequency weights must contain only finite values.",
        )
        value = _check_or_error_if(value, ~jnp.all(value >= 0.0), "frequency weights must be nonnegative.")
        value = _check_or_error_if(
            value,
            ~jnp.any(value > 0.0),
            "frequency weights must contain at least one positive value.",
        )
        self.value = value

    def fit_multiplier(self) -> Array:
        return self.value

    def objective_multiplier(self) -> Array:
        return self.value

    def effective_n(self, n_rows: int) -> Array:
        del n_rows
        return jnp.sum(self.value)


def weights(
    *,
    freq: ArrayLike | None = None,
    var: ArrayLike | None = None,
) -> AbstractWeights:
    r"""Construct semantic sample weights for `fit(..., weights=...)`.

    Use this constructor instead of instantiating weight classes directly.
    Frequency weights are currently supported. Variance weights and combined
    frequency/variance weights are reserved for a later design pass.

    **Arguments:**

    - `freq`: optional frequency weights, shape `(n,)`. `freq[i]` means row
      `i` represents repeated observations.
    - `var`: reserved for variance weights; not implemented yet.

    **Returns:**

    Opaque semantic weight object accepted by `glmax.fit`.

    **Raises:**

    - `ValueError`: if no weights are provided, or frequency weights are not
      rank-1, finite, nonnegative, and at least one positive value.
    - `NotImplementedError`: if `var` is provided.
    """
    if freq is None and var is None:
        raise ValueError("weights(...) requires at least one of `freq` or `var`.")
    if var is not None:
        raise NotImplementedError(
            "variance weights are not implemented yet; use weights(freq=...) for frequency weights."
        )
    return FrequencyWeights(freq)
