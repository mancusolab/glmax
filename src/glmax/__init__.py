# pattern: Imperative Shell

from importlib.metadata import version  # pragma: no cover

import jax

from ._fit import (
    fit as fit,
    FitResult as FitResult,
    FittedGLM as FittedGLM,
    Fitter as Fitter,
    Params as Params,
    predict as predict,
)
from ._infer import (
    AbstractStdErrEstimator as AbstractStdErrEstimator,
    AbstractTest as AbstractInferrer,
    AbstractTest as AbstractTest,
    check as check,
    FisherInfoError as FisherInfoError,
    HuberError as HuberError,
    infer as infer,
    InferenceResult as InferenceResult,
    ScoreTest as ScoreInferrer,
    ScoreTest as ScoreTest,
    WaldTest as WaldInferrer,
    WaldTest as WaldTest,
)
from .data import GLMData as GLMData
from .diagnostics import Diagnostics as Diagnostics
from .glm import GLM as GLM, specify as specify


jax.config.update("jax_enable_x64", True)  # noqa: E402

__version__ = version("glmax")

__all__ = [
    "GLMData",
    "Params",
    "GLM",
    "Fitter",
    "FitResult",
    "FittedGLM",
    "InferenceResult",
    "Diagnostics",
    "AbstractInferrer",
    "WaldInferrer",
    "ScoreInferrer",
    "AbstractStdErrEstimator",
    "FisherInfoError",
    "HuberError",
    "specify",
    "predict",
    "fit",
    "infer",
    "check",
]
