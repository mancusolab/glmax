import pytest

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, FitResult, GLMData, Params


def _make_fit_result() -> FitResult:
    return FitResult(
        params=Params(beta=jnp.array([1.0]), disp=jnp.array(0.0)),
        se=jnp.array([0.5]),
        z=jnp.array([2.0]),
        p=jnp.array([0.05]),
        eta=jnp.array([1.0, 1.0]),
        mu=jnp.array([1.0, 1.0]),
        glm_wt=jnp.array([1.0, 1.0]),
        diagnostics=Diagnostics(
            converged=jnp.array(True),
            num_iters=jnp.array(1),
            objective=jnp.array(0.0),
            objective_delta=jnp.array(-1e-4),
        ),
        curvature=jnp.array([[1.0]]),
        score_residual=jnp.array([0.0, 0.0]),
    )


def test_fit_passes_grammar_nouns_to_custom_fitter() -> None:
    seen: dict[str, object] = {}
    expected = _make_fit_result()

    class RecordingFitter:
        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            seen["model"] = model
            seen["data"] = data
            seen["init"] = init
            return expected

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))
    init = Params(beta=jnp.zeros(1), disp=jnp.array(0.0))

    result = glmax.fit(model, data, init=init, fitter=RecordingFitter())

    assert result is expected
    assert seen["model"] is model
    assert seen["data"] is data
    assert seen["init"] is init


def test_fit_rejects_non_fitresult_from_custom_fitter() -> None:
    class BadFitter:
        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> object:
            del model, data, init
            return object()

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))

    with pytest.raises(TypeError, match="FitResult"):
        glmax.fit(model, data, fitter=BadFitter())
