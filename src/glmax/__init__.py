# pattern: Imperative Shell

from importlib.metadata import version  # pragma: no cover

import jax

from ._fit import (
    AbstractFitter as AbstractFitter,
    fit as fit,
    FitResult as FitResult,
    FittedGLM as FittedGLM,
    Params as Params,
    predict as predict,
)
from ._infer import (
    AbstractStdErrEstimator as AbstractStdErrEstimator,
    AbstractTest as AbstractTest,
    FisherInfoError as FisherInfoError,
    HuberError as HuberError,
    infer as infer,
    InferenceResult as InferenceResult,
    ScoreTest as ScoreTest,
    WaldTest as WaldTest,
)
from .data import GLMData as GLMData
from .diagnostics import check as check, Diagnostics as Diagnostics
from .glm import GLM as GLM, specify as specify


jax.config.update("jax_enable_x64", True)  # noqa: E402

__version__ = version("glmax")

__all__ = [
    "GLMData",
    "Params",
    "GLM",
    "AbstractFitter",
    "FitResult",
    "FittedGLM",
    "InferenceResult",
    "Diagnostics",
    "AbstractTest",
    "WaldTest",
    "ScoreTest",
    "AbstractStdErrEstimator",
    "FisherInfoError",
    "HuberError",
    "specify",
    "predict",
    "fit",
    "infer",
    "check",
]
