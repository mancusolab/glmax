import importlib

from dataclasses import replace

import pytest

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, GLMData
from glmax.family import Gaussian


def _make_fit_result():
    model = glmax.GLM(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        y=jnp.array([1.2, 1.9, 3.1, 4.2]),
    )
    fit_result = glmax.fit(model, data)
    return model, fit_result


def test_check_returns_diagnostics_without_refitting() -> None:
    model, fit_result = _make_fit_result()

    diagnostics = glmax.check(model, fit_result)

    assert isinstance(diagnostics, Diagnostics)
    assert diagnostics is fit_result.diagnostics


def test_check_rejects_invalid_model_and_result_contracts() -> None:
    model, fit_result = _make_fit_result()

    with pytest.raises(TypeError, match="GLM"):
        glmax.check(object(), fit_result)

    with pytest.raises(TypeError, match="FitResult"):
        glmax.check(model, object())


def test_check_rejects_invalid_fit_artifacts_deterministically() -> None:
    model, fit_result = _make_fit_result()

    bad_result = replace(
        fit_result,
        diagnostics=replace(
            fit_result.diagnostics,
            objective=jnp.array(jnp.nan),
        ),
    )
    with pytest.raises(ValueError, match="FitResult.diagnostics.objective"):
        glmax.check(model, bad_result)

    with pytest.raises(TypeError, match="FitResult.params.beta must be numeric"):
        glmax.check(
            model,
            replace(
                fit_result,
                params=glmax.Params(beta=["bad"], disp=jnp.array(0.0)),
            ),
        )


def test_check_never_calls_fit_or_irls(monkeypatch: pytest.MonkeyPatch) -> None:
    model, fit_result = _make_fit_result()

    def fail_fit(*_args, **_kwargs):
        raise AssertionError("GLM.fit should never be called by check(...).")

    def fail_irls(*_args, **_kwargs):
        raise AssertionError("infer.optimize.irls should never be called by check(...).")

    monkeypatch.setattr(glmax.GLM, "fit", fail_fit)
    infer_optimize = importlib.import_module("glmax.infer.optimize")
    monkeypatch.setattr(infer_optimize, "irls", fail_irls)

    diagnostics = glmax.check(model, fit_result)
    assert isinstance(diagnostics, Diagnostics)
