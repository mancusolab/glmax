# pattern: Imperative Shell

import warnings

from .inference import (
    AbstractStdErrEstimator as AbstractStdErrEstimator,
    FisherInfoError as FisherInfoError,
    HuberError as HuberError,
)


warnings.warn(
    "`glmax.infer.stderr` is deprecated; import error estimators from `glmax.infer.inference` instead.",
    DeprecationWarning,
    stacklevel=2,
)
