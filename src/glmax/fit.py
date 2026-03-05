from __future__ import annotations

from .contracts import FitResult, Fitter, GLMData, Params
from .glm import GLM


class _ModelFitter:
    """Bridge canonical fit verb calls into the existing GLM.fit implementation."""

    def __call__(self, model: GLM, data: GLMData, init: Params | None = None) -> FitResult:
        offset = 0.0 if data.offset is None else data.offset
        if init is None:
            return model.fit(data.X, data.y, offset_eta=offset)

        return model.fit(data.X, data.y, offset_eta=offset, init=init.beta, alpha_init=init.disp)


DEFAULT_FITTER: Fitter = _ModelFitter()


def fit(model: GLM, data: GLMData, init: Params | None = None, *, fitter: Fitter = DEFAULT_FITTER) -> FitResult:
    """Canonical fit verb surface."""
    return fitter(model, data, init)
