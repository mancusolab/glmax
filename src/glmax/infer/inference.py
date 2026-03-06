# pattern: Functional Core

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from ..contracts import FitResult, InferenceResult, validate_fit_result


if TYPE_CHECKING:
    from ..glm import GLM


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
    p = model.wald_test(z, df)

    return InferenceResult(params=fit_result.params, se=se, z=z, p=p)
