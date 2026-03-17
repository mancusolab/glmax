# pattern: Imperative Shell

from dataclasses import fields

import pytest

import jax.numpy as jnp

import glmax

from glmax import FittedGLM, GLMData, InferenceResult
from glmax._infer.hyptest import ScoreTest, WaldTest
from glmax._infer.stderr import AbstractStdErrEstimator, HuberError
from glmax.family import Gaussian


def unchecked_fit_result(base, **overrides: object):
    values = {field.name: getattr(base, field.name) for field in fields(type(base))}
    values.update(overrides)

    result = object.__new__(type(base))
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def unchecked_fitted(base: FittedGLM, **overrides: object) -> FittedGLM:
    values = {"model": base.model, "result": base.result}
    values.update(overrides)

    fitted = object.__new__(type(base))
    for name, value in values.items():
        object.__setattr__(fitted, name, value)
    return fitted


def _make_fitted():
    model = glmax.specify(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        y=jnp.array([1.2, 1.9, 3.1, 4.2]),
    )
    fitted = glmax.fit(model, data)
    return fitted


def test_infer_returns_inference_result_without_refitting() -> None:
    fitted = _make_fitted()

    inferred = glmax.infer(fitted)

    assert isinstance(inferred, InferenceResult)
    assert inferred.params is fitted.params
    assert inferred.se.shape == fitted.params.beta.shape
    assert inferred.stat.shape == fitted.params.beta.shape
    assert inferred.p.shape == fitted.params.beta.shape
    with pytest.raises(AttributeError):
        inferred.z


def test_infer_uses_injected_stderr_estimator() -> None:
    fitted = _make_fitted()

    class RecordingStdErr(AbstractStdErrEstimator):
        def __call__(self, fitted_arg):
            assert fitted_arg is fitted
            return jnp.array([[4.0]])

    inferred = glmax.infer(fitted, stderr=RecordingStdErr())

    assert jnp.allclose(inferred.se, jnp.array([2.0]))


def test_infer_rejects_invalid_model_and_result_contracts() -> None:
    fitted = _make_fitted()

    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.infer(object())

    with pytest.raises(AttributeError):
        glmax.infer(unchecked_fitted(fitted, result=object()))

    with pytest.raises(TypeError, match="AbstractStdErrEstimator"):
        glmax.infer(fitted, stderr=object())

    with pytest.raises(TypeError, match="AbstractTest"):
        glmax.infer(fitted, inferrer=object())


def test_infer_default_matches_explicit_wald_test() -> None:
    fitted = _make_fitted()

    result_default = glmax.infer(fitted)
    result_explicit = glmax.infer(fitted, inferrer=WaldTest())

    assert jnp.allclose(result_default.stat, result_explicit.stat)
    assert jnp.allclose(result_default.se, result_explicit.se)
    assert jnp.allclose(result_default.p, result_explicit.p)


def test_infer_routes_to_score_test_when_specified() -> None:
    fitted = _make_fitted()

    result = glmax.infer(fitted, inferrer=ScoreTest())

    assert isinstance(result, InferenceResult)
    assert jnp.all(jnp.isnan(result.se))
    assert jnp.all(jnp.isfinite(result.stat))
    assert jnp.all((result.p >= 0.0) & (result.p <= 1.0))
    assert result.stat.shape == fitted.params.beta.shape


def test_infer_passes_stderr_into_inferrer() -> None:
    fitted = _make_fitted()
    call_count = {"n": 0}

    class CountingStdErr(AbstractStdErrEstimator):
        def __call__(self, fitted_arg):
            call_count["n"] += 1
            return jnp.array([[4.0]])

    result = glmax.infer(fitted, stderr=CountingStdErr())

    assert call_count["n"] == 1
    assert result.se.shape == fitted.params.beta.shape
    assert jnp.allclose(result.se, jnp.array([2.0]))


def test_infer_accepts_huber_error() -> None:
    fitted = _make_fitted()

    result = glmax.infer(fitted, stderr=HuberError())

    assert isinstance(result, InferenceResult)
    assert result.se.shape == fitted.params.beta.shape


def test_inferrer_types_importable_from_glmax() -> None:
    from glmax import AbstractTest, ScoreTest, WaldTest  # noqa: F401

    assert AbstractTest is not None
    assert WaldTest is not None
    assert ScoreTest is not None


def test_stderr_types_importable_from_glmax() -> None:
    from glmax import AbstractStdErrEstimator, FisherInfoError, HuberError  # noqa: F401

    assert AbstractStdErrEstimator is not None
    assert FisherInfoError is not None
    assert HuberError is not None
