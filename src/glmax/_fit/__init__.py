"""Internal fit kernels."""

from .fit import (
    fit as fit,
    predict as predict,
)
from .irls import IRLSFitter as IRLSFitter
from .newton import NewtonFitter as NewtonFitter
from .types import (
    AbstractFitter as AbstractFitter,
    FitResult as FitResult,
    FittedGLM as FittedGLM,
    Params as Params,
)


__all__ = [
    "Params",
    "FitResult",
    "FittedGLM",
    "AbstractFitter",
    "IRLSFitter",
    "fit",
    "predict",
]
