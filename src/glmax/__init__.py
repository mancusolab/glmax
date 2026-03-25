# pattern: Imperative Shell

from importlib.metadata import version  # pragma: no cover

import jax

from ._fit import (
    AbstractFitter as AbstractFitter,
    fit as fit,
    FitResult as FitResult,
    FittedGLM as FittedGLM,
    IRLSFitter as IRLSFitter,
    NewtonFitter as NewtonFitter,
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
from .diagnostics import (
    AbstractDiagnostic as AbstractDiagnostic,
    check as check,
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
    CauchitLink as CauchitLink,
    CLogLogLink as CLogLogLink,
    ExponentialDispersionFamily as ExponentialDispersionFamily,
    Gamma as Gamma,
    Gaussian as Gaussian,
    IdentityLink as IdentityLink,
    InverseGaussian as InverseGaussian,
    InverseLink as InverseLink,
    LogitLink as LogitLink,
    LogLink as LogLink,
    LogLogLink as LogLogLink,
    NBLink as NBLink,
    NegativeBinomial as NegativeBinomial,
    Poisson as Poisson,
    PowerLink as PowerLink,
    ProbitLink as ProbitLink,
    SqrtLink as SqrtLink,
)


jax.config.update("jax_enable_x64", True)  # noqa: E402

__version__ = version("glmax")
