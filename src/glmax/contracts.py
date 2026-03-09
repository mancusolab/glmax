# pattern: Functional Core

"""Compatibility re-exports for legacy contract imports."""

from .data import GLMData
from .fit import FitResult, Fitter, Params, validate_fit_result
from .infer.diagnostics import Diagnostics
from .infer.inference import InferenceResult


__all__ = ["GLMData", "Params", "Diagnostics", "FitResult", "InferenceResult", "Fitter", "validate_fit_result"]
