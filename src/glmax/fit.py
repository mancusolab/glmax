from jaxtyping import ArrayLike

from .glm import GLM, GLMState


def fit(
    model: GLM,
    X: ArrayLike,
    y: ArrayLike,
    offset: ArrayLike | None = None,
    *,
    fitter: object | None = None,
    solver: object | None = None,
    covariance: object | None = None,
    tests: object | None = None,
    init: ArrayLike | None = None,
    options: dict[str, object] | None = None,
) -> GLMState:
    del fitter, covariance, tests

    if solver is not None:
        model = GLM(family=model.family, solver=solver)

    if offset is None:
        offset = y * 0.0

    fit_options = options or {}
    return model.fit(X, y, offset_eta=offset, init=init, **fit_options)
