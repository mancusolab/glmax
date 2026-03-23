# pattern: Imperative Shell

import importlib

from dataclasses import fields

import pytest

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

import glmax

from glmax import (
    AbstractDiagnostic,
    FitResult,
    FittedGLM,
    GofStats,
    InferenceResult,
    InfluenceStats,
    Params,
)
from glmax.family import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson


DIAGNOSTICS = (
    glmax.PearsonResidual(),
    glmax.DevianceResidual(),
    glmax.QuantileResidual(),
    glmax.GoodnessOfFit(),
    glmax.Influence(),
)


def unchecked_fit_result(base: FitResult, **overrides: object) -> FitResult:
    values = {field.name: getattr(base, field.name) for field in fields(type(base))}
    values.update(overrides)

    result = object.__new__(FitResult)
    for name, value in values.items():
        object.__setattr__(result, name, value)
    return result


def unchecked_fitted(base: FittedGLM, **overrides: object) -> FittedGLM:
    values = {"family": base.family, "result": base.result}
    values.update(overrides)

    fitted = object.__new__(FittedGLM)
    for name, value in values.items():
        object.__setattr__(fitted, name, value)
    return fitted


def _assert_canonical_params_for_family(family, params: Params) -> None:
    assert params._fields == ("beta", "disp", "aux")

    if isinstance(family, NegativeBinomial):
        assert jnp.allclose(params.disp, jnp.array(1.0))
        assert params.aux is not None
        assert float(jnp.asarray(params.aux)) > 0.0
        return

    assert params.aux is None
    if isinstance(family, (Gaussian, Gamma)):
        assert float(jnp.asarray(params.disp)) > 0.0
    else:
        assert jnp.allclose(params.disp, jnp.array(1.0))


@pytest.mark.parametrize(
    ("family", "X", "y"),
    [
        (Gaussian(), jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), jnp.array([0.1, 1.0, 2.1, 2.9, 4.2])),
        (Gamma(), jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]]), jnp.array([0.8, 1.1, 1.7, 2.2, 2.9])),
        (Poisson(), jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), jnp.array([0.0, 1.0, 1.0, 2.0, 3.0])),
        (Binomial(), jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])),
        (NegativeBinomial(), jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), jnp.array([0.0, 1.0, 2.0, 1.0, 4.0])),
    ],
)
def test_grammar_contract_matrix_across_all_verbs(family, X, y) -> None:
    current_fitted_glm_type = importlib.import_module("glmax._fit").FittedGLM

    fitted = glmax.fit(family, X, y)
    prediction = glmax.predict(family, fitted.params, X)
    inferred = glmax.infer(fitted)
    default_diagnostic = glmax.check(fitted)
    diagnostics = jtu.tree_map(
        lambda diagnostic: glmax.check(fitted, diagnostic=diagnostic),
        DIAGNOSTICS,
        is_leaf=lambda node: isinstance(node, AbstractDiagnostic),
    )

    assert isinstance(fitted, current_fitted_glm_type)
    _assert_canonical_params_for_family(family, fitted.params)
    assert prediction.shape == y.shape
    assert isinstance(inferred, InferenceResult)
    _assert_canonical_params_for_family(family, inferred.params)
    assert inferred.se.shape == fitted.params.beta.shape
    assert isinstance(default_diagnostic, GofStats)
    assert isinstance(diagnostics, tuple)
    assert len(diagnostics) == len(DIAGNOSTICS)
    pearson, deviance, quantile, gof, influence = diagnostics
    assert pearson.shape == y.shape
    assert deviance.shape == y.shape
    assert quantile.shape == y.shape
    assert isinstance(gof, GofStats)
    assert isinstance(influence, InfluenceStats)


def test_grammar_contract_matrix_rejects_invalid_noun_usage() -> None:
    family = Gaussian()
    X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    y = jnp.array([0.1, 1.2, 1.8, 3.1])
    fitted = glmax.fit(family, X, y)

    with pytest.raises(TypeError, match="ExponentialDispersionFamily"):
        glmax.fit(object(), X, y)

    with pytest.raises(TypeError, match="Params"):
        glmax.predict(family, jnp.array([1.0]), X)

    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.infer(object())

    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.check(object())

    inferred = glmax.infer(
        unchecked_fitted(
            fitted,
            result=unchecked_fit_result(
                fitted.result,
                params=Params(beta=jnp.array([jnp.nan]), disp=jnp.array(1.0), aux=None),
            ),
        ),
    )
    assert isinstance(inferred, InferenceResult)
    assert bool(jnp.isnan(inferred.stat).any() or jnp.isnan(inferred.p).any())

    with pytest.raises(eqx.EquinoxRuntimeError, match="fitted.params.disp"):
        glmax.infer(
            unchecked_fitted(
                fitted,
                result=unchecked_fit_result(
                    fitted.result,
                    params=Params(beta=jnp.array([1], dtype=jnp.int32), disp=jnp.array(0.0), aux=None),
                ),
            ),
        )
