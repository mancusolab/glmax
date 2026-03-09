# pattern: Imperative Shell

from dataclasses import fields

import pytest

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, FitResult, GLMData, Params
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


def unchecked_fit_result(base: FitResult, **overrides: object) -> FitResult:
    values = {field.name: getattr(base, field.name) for field in fields(type(base))}
    values.update(overrides)

    result = object.__new__(FitResult)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


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


@pytest.mark.parametrize(
    ("family", "y"),
    [
        (Gaussian(), jnp.array([0.2, 1.1, 2.0, 3.3])),
        (Poisson(), jnp.array([0.0, 1.0, 1.0, 2.0])),
        (Binomial(), jnp.array([0.0, 1.0, 0.0, 1.0])),
        (NegativeBinomial(), jnp.array([0.0, 1.0, 2.0, 4.0])),
    ],
)
def test_default_fitter_returns_canonical_fitresult_for_supported_families(family, y) -> None:
    model = glmax.specify(family=family)
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=y)

    result = glmax.fit(model, data)

    assert isinstance(result, FitResult)
    assert result.params.beta.shape == (1,)
    assert result.eta.shape == y.shape
    assert result.mu.shape == y.shape
    assert result.glm_wt.shape == y.shape
    assert result.score_residual.shape == y.shape


def test_fit_rejects_custom_fitter_result_with_malformed_contract() -> None:
    malformed = unchecked_fit_result(_make_fit_result(), diagnostics=object())

    class BadFitter:
        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            del model, data, init
            return malformed

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))

    with pytest.raises(TypeError, match="FitResult.diagnostics"):
        glmax.fit(model, data, fitter=BadFitter())
