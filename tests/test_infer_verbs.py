# pattern: Imperative Shell

import importlib

from dataclasses import fields

import pytest

import jax.numpy as jnp

import glmax

from glmax import GLMData, InferenceResult, Params
from glmax.family import Gaussian


def unchecked_fit_result(base, **overrides: object):
    values = {field.name: getattr(base, field.name) for field in fields(type(base))}
    values.update(overrides)

    result = object.__new__(type(base))
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def _make_fit_result():
    model = glmax.specify(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        y=jnp.array([1.2, 1.9, 3.1, 4.2]),
    )
    fit_result = glmax.fit(model, data)
    return model, fit_result


def test_infer_returns_inference_result_without_refitting() -> None:
    model, fit_result = _make_fit_result()

    inferred = glmax.infer(model, fit_result)

    assert isinstance(inferred, InferenceResult)
    assert inferred.params is fit_result.params
    assert jnp.allclose(inferred.se, fit_result.se)
    assert jnp.allclose(inferred.z, fit_result.z)
    assert jnp.allclose(inferred.p, fit_result.p)


def test_infer_rejects_invalid_model_and_result_contracts() -> None:
    model, fit_result = _make_fit_result()

    with pytest.raises(TypeError, match="GLM"):
        glmax.infer(object(), fit_result)

    with pytest.raises(TypeError, match="FitResult"):
        glmax.infer(model, object())


def test_infer_rejects_invalid_fit_artifacts_deterministically() -> None:
    model, fit_result = _make_fit_result()

    with pytest.raises(ValueError, match="FitResult.curvature"):
        glmax.infer(model, unchecked_fit_result(fit_result, curvature=jnp.ones((2, 1))))

    with pytest.raises(ValueError, match="FitResult.params.beta"):
        glmax.infer(
            model,
            unchecked_fit_result(fit_result, params=Params(beta=jnp.array([1.0, 2.0]), disp=jnp.array(0.0))),
        )

    with pytest.raises(ValueError, match="FitResult.params.beta"):
        glmax.infer(
            model,
            unchecked_fit_result(fit_result, params=Params(beta=jnp.array([jnp.nan]), disp=jnp.array(0.0))),
        )

    with pytest.raises(TypeError, match="FitResult.params.beta must be numeric"):
        glmax.infer(
            model,
            unchecked_fit_result(fit_result, params=Params(beta=["bad"], disp=jnp.array(0.0))),
        )

    with pytest.raises(TypeError, match="FitResult.params.disp must be numeric"):
        glmax.infer(
            model,
            unchecked_fit_result(fit_result, params=Params(beta=jnp.array([1.0]), disp="bad")),
        )

    with pytest.raises(ValueError, match="FitResult.params.disp"):
        glmax.infer(
            model,
            unchecked_fit_result(fit_result, params=Params(beta=jnp.array([1.0]), disp=jnp.array(jnp.inf))),
        )

    with pytest.raises(TypeError, match="FitResult.params.beta must have an inexact dtype"):
        glmax.infer(
            model,
            unchecked_fit_result(fit_result, params=Params(beta=jnp.array([1], dtype=jnp.int32), disp=jnp.array(0.0))),
        )

    with pytest.raises(TypeError, match="FitResult.params.disp must have an inexact dtype"):
        glmax.infer(
            model,
            unchecked_fit_result(fit_result, params=Params(beta=jnp.array([1.0]), disp=jnp.array(0, dtype=jnp.int32))),
        )


def test_infer_never_calls_fit_or_irls(monkeypatch: pytest.MonkeyPatch) -> None:
    model, fit_result = _make_fit_result()

    def fail_fit(*_args, **_kwargs):
        raise AssertionError("GLM.fit should never be called by infer(...).")

    def fail_irls(*_args, **_kwargs):
        raise AssertionError("infer.optimize.irls should never be called by infer(...).")

    monkeypatch.setattr(glmax.GLM, "fit", fail_fit)

    infer_optimize = importlib.import_module("glmax.infer.optimize")
    monkeypatch.setattr(infer_optimize, "irls", fail_irls)

    inferred = glmax.infer(model, fit_result)
    assert isinstance(inferred, InferenceResult)
