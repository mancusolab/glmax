# pattern: Functional Core

from __future__ import annotations

from typing import NamedTuple

from ._fit import FittedGLM


__all__ = ["Diagnostics", "check"]


class Diagnostics(NamedTuple):
    """Placeholder model-fit diagnostics contract returned by check()."""


def check(fitted: FittedGLM) -> Diagnostics:
    """Placeholder model-fit assessment seam over fit artifacts."""
    if not isinstance(fitted, FittedGLM):
        raise TypeError("check(...) expects `fitted` to be a FittedGLM instance.")

    return Diagnostics()
