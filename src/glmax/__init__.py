# pattern: Imperative Shell

from importlib.metadata import version  # pragma: no cover

import jax

from .data import GLMData as GLMData
from .fit import fit as fit, FitResult as FitResult, Fitter as Fitter, Params as Params, predict as predict
from .glm import GLM as GLM, specify as specify
from .infer import check as check, infer as infer
from .infer.diagnostics import Diagnostics as Diagnostics
from .infer.inference import InferenceResult as InferenceResult


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
