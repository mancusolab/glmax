# pattern: Imperative Shell

from dataclasses import replace

import pytest

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, FitResult, GLMData, InferenceResult, Params
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


@pytest.mark.parametrize(
    ("family", "y"),
    [
        (Gaussian(), jnp.array([0.1, 1.0, 2.1, 2.9, 4.2])),
        (Poisson(), jnp.array([0.0, 1.0, 1.0, 2.0, 3.0])),
        (Binomial(), jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])),
        (NegativeBinomial(), jnp.array([0.0, 1.0, 2.0, 1.0, 4.0])),
    ],
)
def test_grammar_contract_matrix_across_all_verbs(family, y) -> None:
    model = glmax.specify(family=family)
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), y=y)

    fit_result = glmax.fit(model, data)
    prediction = glmax.predict(model, fit_result.params, data)
    inferred = glmax.infer(model, fit_result)
    diagnostics = glmax.check(model, fit_result)

    assert isinstance(fit_result, FitResult)
    assert prediction.shape == y.shape
    assert isinstance(inferred, InferenceResult)
    assert inferred.se.shape == fit_result.se.shape
    assert isinstance(diagnostics, Diagnostics)
    assert bool(jnp.isfinite(diagnostics.objective))


def test_grammar_contract_matrix_rejects_invalid_noun_usage() -> None:
    model = glmax.specify(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.1, 1.2, 1.8, 3.1]))
    fit_result = glmax.fit(model, data)

    with pytest.raises(TypeError, match="GLM"):
        glmax.fit(object(), data)

    with pytest.raises(TypeError, match="GLMData"):
        glmax.fit(model, jnp.ones((4, 1)))

    with pytest.raises(TypeError, match="Params"):
        glmax.predict(model, jnp.array([1.0]), data)

    with pytest.raises(TypeError, match="FitResult"):
        glmax.infer(model, object())

    with pytest.raises(TypeError, match="FitResult"):
        glmax.check(model, object())

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.predict(model, Params(beta=jnp.array([jnp.nan]), disp=jnp.array(0.0)), data)

    with pytest.raises(ValueError, match="FitResult.params.beta"):
        glmax.infer(
            model,
            replace(fit_result, params=Params(beta=jnp.array([jnp.nan]), disp=jnp.array(0.0))),
        )

    with pytest.raises(TypeError, match="Params.beta must have an inexact dtype"):
        glmax.predict(model, Params(beta=jnp.array([1], dtype=jnp.int32), disp=jnp.array(0.0)), data)

    with pytest.raises(TypeError, match="FitResult.params.beta must have an inexact dtype"):
        glmax.infer(
            model,
            replace(fit_result, params=Params(beta=jnp.array([1], dtype=jnp.int32), disp=jnp.array(0.0))),
        )
