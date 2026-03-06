# pattern: Imperative Shell

from importlib.metadata import version  # pragma: no cover

import jax

from .contracts import (
    Diagnostics as Diagnostics,
    FitResult as FitResult,
    Fitter as Fitter,
    GLMData as GLMData,
    InferenceResult as InferenceResult,
    Params as Params,
)
from .fit import fit as fit, predict as predict
from .glm import GLM as GLM, specify as specify
from .infer import check as check, infer as infer


jax.config.update("jax_enable_x64", True)  # noqa: E402

__version__ = version("glmax")

__all__ = [
    "GLMData",
    "Params",
    "GLM",
    "Fitter",
    "FitResult",
    "InferenceResult",
    "Diagnostics",
    "specify",
    "predict",
    "fit",
    "infer",
    "check",
]
