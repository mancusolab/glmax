# pattern: Imperative Shell

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..fit import FittedGLM
    from .diagnostics import Diagnostics
    from .inference import InferenceResult
    from .stderr import AbstractStdErrEstimator


def infer(fitted: "FittedGLM", stderr: "AbstractStdErrEstimator" = None) -> "InferenceResult":
    """Canonical infer verb entrypoint."""
    from .inference import infer as _infer

    if stderr is None:
        return _infer(fitted)
    return _infer(fitted, stderr)


def check(fitted: "FittedGLM") -> "Diagnostics":
    """Canonical check verb entrypoint."""
    from .diagnostics import check as _check

    return _check(fitted)


__all__ = ["infer", "check"]
