# pattern: Functional Core

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp

from jax import Array
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from ..family.dist import ExponentialFamily, Gaussian
from ..family.utils import t_cdf
from ..fit import FitResult, Params, validate_fit_result


if TYPE_CHECKING:
    from ..glm import GLM


__all__ = ["InferenceResult", "infer", "wald_test"]


class InferenceResult(NamedTuple):
    """Canonical infer verb output contract."""

    params: Params
    se: Array
    z: Array
    p: Array


def wald_test(statistic: ArrayLike, df: int, family: ExponentialFamily) -> Array:
    r"""Two-sided Wald test p-values.

    Uses a $t_{df}$ distribution for Gaussian families and $\mathcal{N}(0, 1)$
    for all others.

    **Arguments:**

    - `statistic`: test statistics $\hat\beta / \mathrm{SE}(\hat\beta)$, shape `(p,)`.
    - `df`: residual degrees of freedom $n - p$.
    - `family`: fitted `ExponentialFamily` instance.

    **Returns:**

    Two-sided p-values, shape `(p,)`.
    """
    if isinstance(family, Gaussian):
        return 2 * t_cdf(-jnp.abs(statistic), df)
    return 2 * norm.sf(jnp.abs(statistic))


def infer(model: GLM, fit_result: FitResult) -> InferenceResult:
    """Inferential summaries from fit artifacts without refitting."""
    from ..glm import GLM as _GLM

    if not isinstance(model, _GLM):
        raise TypeError("infer(...) expects `model` to be a GLM instance.")
    if not isinstance(fit_result, FitResult):
        raise TypeError("infer(...) expects `fit_result` to be a FitResult instance.")

    validate_fit_result(fit_result)

    beta = jnp.asarray(fit_result.params.beta)
    curvature = jnp.asarray(fit_result.curvature)
    se = jnp.sqrt(jnp.diag(curvature))
    z = beta / se
    df = int(fit_result.eta.shape[0] - beta.shape[0])
    p = wald_test(z, df, model.family)

    return InferenceResult(params=fit_result.params, se=se, z=z, p=p)
