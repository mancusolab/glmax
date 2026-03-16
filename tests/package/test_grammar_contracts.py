# pattern: Imperative Shell

import importlib

from dataclasses import fields

import pytest

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, FitResult, FittedGLM, GLMData, InferenceResult, Params
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


def unchecked_fit_result(base: FitResult, **overrides: object) -> FitResult:
    values = {field.name: getattr(base, field.name) for field in fields(type(base))}
    values.update(overrides)

    result = object.__new__(FitResult)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def unchecked_fitted(base: FittedGLM, **overrides: object) -> FittedGLM:
    values = {"model": base.model, "result": base.result}
    values.update(overrides)

    fitted = object.__new__(FittedGLM)
    for name, value in values.items():
        object.__setattr__(fitted, name, value)
    return fitted


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
    current_fitted_glm_type = importlib.import_module("glmax._fit").FittedGLM
    model = glmax.specify(family=family)
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), y=y)

    fitted = glmax.fit(model, data)
    prediction = glmax.predict(model, fitted.params, data)
    inferred = glmax.infer(fitted)
    diagnostics = glmax.check(fitted)

    assert isinstance(fitted, current_fitted_glm_type)
    assert prediction.shape == y.shape
    assert isinstance(inferred, InferenceResult)
    assert inferred.se.shape == fitted.params.beta.shape
    assert isinstance(diagnostics, Diagnostics)
    assert diagnostics == Diagnostics()


def test_grammar_contract_matrix_rejects_invalid_noun_usage() -> None:
    model = glmax.specify(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.1, 1.2, 1.8, 3.1]))
    fitted = glmax.fit(model, data)

    with pytest.raises(TypeError, match="GLM"):
        glmax.fit(object(), data)

    with pytest.raises(TypeError, match="GLMData"):
        glmax.fit(model, jnp.ones((4, 1)))

    with pytest.raises(TypeError, match="Params"):
        glmax.predict(model, jnp.array([1.0]), data)

    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.infer(object())

    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.check(object())

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.predict(model, Params(beta=jnp.array([jnp.nan]), disp=jnp.array(0.0)), data)

    inferred = glmax.infer(
        unchecked_fitted(
            fitted,
            result=unchecked_fit_result(fitted.result, params=Params(beta=jnp.array([jnp.nan]), disp=jnp.array(0.0))),
        ),
    )
    assert isinstance(inferred, InferenceResult)
    assert bool(jnp.isnan(inferred.stat).any() or jnp.isnan(inferred.p).any())

    with pytest.raises(TypeError, match="Params.beta must have an inexact dtype"):
        glmax.predict(model, Params(beta=jnp.array([1], dtype=jnp.int32), disp=jnp.array(0.0)), data)

    inferred = glmax.infer(
        unchecked_fitted(
            fitted,
            result=unchecked_fit_result(
                fitted.result,
                params=Params(beta=jnp.array([1], dtype=jnp.int32), disp=jnp.array(0.0)),
            ),
        ),
    )
    assert isinstance(inferred, InferenceResult)


def test_dead_modules_are_not_importable() -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax.contracts")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax.predict")  # the stub module, not the verb


def test_wald_test_importable_from_infer_inference() -> None:
    from glmax._infer.hyptest import _wald_test

    assert callable(_wald_test)
