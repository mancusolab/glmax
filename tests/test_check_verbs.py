# pattern: Imperative Shell

import importlib

from dataclasses import fields

import pytest

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, FittedGLM, GLMData
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
    return glmax.fit(model, data)


def test_check_returns_diagnostics_without_refitting() -> None:
    fitted = _make_fitted()

    diagnostics = glmax.check(fitted)

    assert isinstance(diagnostics, Diagnostics)
    assert diagnostics == Diagnostics()


def test_check_rejects_invalid_model_and_result_contracts() -> None:
    fitted = _make_fitted()

    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.check(object())

    with pytest.raises(TypeError, match="FitResult"):
        glmax.check(unchecked_fitted(fitted, result=object()))


def test_check_rejects_invalid_fit_artifacts_deterministically() -> None:
    fitted = _make_fitted()
    fit_result = fitted.result

    bad_result = unchecked_fit_result(
        fit_result,
        objective=jnp.array(jnp.nan),
    )
    with pytest.raises(ValueError, match="FitResult.objective"):
        glmax.check(unchecked_fitted(fitted, result=bad_result))

    with pytest.raises(TypeError, match="FitResult.params.beta must be numeric"):
        glmax.check(
            unchecked_fitted(
                fitted,
                result=unchecked_fit_result(
                    fit_result,
                    params=glmax.Params(beta=["bad"], disp=jnp.array(0.0)),
                ),
            ),
        )

    with pytest.raises(ValueError, match="FitResult.params.disp"):
        glmax.check(
            unchecked_fitted(
                fitted,
                result=unchecked_fit_result(
                    fit_result,
                    params=glmax.Params(beta=jnp.array([1.0]), disp=jnp.array(jnp.inf)),
                ),
            ),
        )


def test_check_never_calls_fit_or_irls(monkeypatch: pytest.MonkeyPatch) -> None:
    fitted = _make_fitted()

    def fail_irls(*_args, **_kwargs):
        raise AssertionError("infer.optimize.irls should never be called by check(...).")

    infer_optimize = importlib.import_module("glmax.infer.optimize")
    monkeypatch.setattr(infer_optimize, "irls", fail_irls)

    diagnostics = glmax.check(fitted)
    assert isinstance(diagnostics, Diagnostics)
