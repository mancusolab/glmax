# pattern: Imperative Shell

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..fit import FittedGLM
    from .diagnostics import Diagnostics
    from .inference import InferenceResult
    from .inferrer import AbstractInferrer
    from .stderr import AbstractStdErrEstimator


def infer(
    fitted: "FittedGLM",
    inferrer: "AbstractInferrer" = None,
    stderr: "AbstractStdErrEstimator" = None,
) -> "InferenceResult":
    """Canonical infer verb entrypoint.

    **Arguments:**

    - `fitted`: fitted model returned by `glmax.fit(...)`.
    - `inferrer`: optional inference strategy to route through.
    - `stderr`: optional standard-error estimator used by compatible inferrers.

    **Returns:**

    - `InferenceResult` from the delegated infer implementation.

    **Raises:**

    - `TypeError`: propagated from the delegated infer boundary when argument
      contracts are violated.
    """
    from .inference import infer as _infer

    kwargs = {}
    if inferrer is not None:
        kwargs["inferrer"] = inferrer
    if stderr is not None:
        kwargs["stderr"] = stderr
    return _infer(fitted, **kwargs)


def check(fitted: "FittedGLM") -> "Diagnostics":
    """Canonical check verb entrypoint."""
    from .diagnostics import check as _check

    return _check(fitted)


__all__ = ["infer", "check"]
