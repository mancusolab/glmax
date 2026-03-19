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
from .diagnostics import (
    AbstractDiagnostic as AbstractDiagnostic,
    check as check,
    DEFAULT_DIAGNOSTICS as DEFAULT_DIAGNOSTICS,
    DevianceResidual as DevianceResidual,
    GofStats as GofStats,
    GoodnessOfFit as GoodnessOfFit,
    Influence as Influence,
    InfluenceStats as InfluenceStats,
    PearsonResidual as PearsonResidual,
    QuantileResidual as QuantileResidual,
)
from .family import (
    AbstractLink as AbstractLink,
    Binomial as Binomial,
    ExponentialDispersionFamily as ExponentialDispersionFamily,
    Gamma as Gamma,
    Gaussian as Gaussian,
    IdentityLink as IdentityLink,
    InverseLink as InverseLink,
    LogitLink as LogitLink,
    LogLink as LogLink,
    NBLink as NBLink,
    NegativeBinomial as NegativeBinomial,
    Poisson as Poisson,
    PowerLink as PowerLink,
)
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
    "AbstractDiagnostic",
    "DEFAULT_DIAGNOSTICS",
    "DevianceResidual",
    "GoodnessOfFit",
    "GofStats",
    "Influence",
    "InfluenceStats",
    "PearsonResidual",
    "QuantileResidual",
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
