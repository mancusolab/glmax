# pattern: Imperative Shell

import pytest

import equinox as eqx
import jax.numpy as jnp

import glmax

from glmax.diagnostics import (
    DevianceResidual,
    GofStats,
    InfluenceStats,
    PearsonResidual,
    QuantileResidual,
)
from glmax.family import Gaussian


def _make_fitted():
    model = glmax.GLM(family=Gaussian())
    X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    y = jnp.array([1.2, 1.9, 3.1, 4.2])
    return glmax.fit(model, X, y)


def test_check_default_returns_pearson_residual():
    fitted = _make_fitted()
    result = glmax.check(fitted)
    assert isinstance(result, GofStats)


def test_check_accepts_single_concrete_diagnostics():
    fitted = _make_fitted()
    pearson = glmax.check(fitted, diagnostic=PearsonResidual())
    deviance = glmax.check(fitted, diagnostic=DevianceResidual())
    quantile = glmax.check(fitted, diagnostic=QuantileResidual())
    gof = glmax.check(fitted, diagnostic=glmax.GoodnessOfFit())
    influence = glmax.check(fitted, diagnostic=glmax.Influence())

    assert isinstance(pearson, jnp.ndarray)
    assert isinstance(deviance, jnp.ndarray)
    assert isinstance(quantile, jnp.ndarray)
    assert isinstance(gof, GofStats)
    assert isinstance(influence, InfluenceStats)


def test_check_custom_diagnostic_single():
    fitted = _make_fitted()
    result = glmax.check(fitted, diagnostic=PearsonResidual())
    assert result.shape == (4,)


def test_check_rejects_non_fitted_glm():
    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.check("not_a_fitted_glm")


def test_check_filter_jit_produces_same_result():
    fitted = _make_fitted()
    result_eager = glmax.check(fitted, diagnostic=PearsonResidual())
    result_jit = eqx.filter_jit(glmax.check)(fitted, diagnostic=PearsonResidual())
    assert jnp.allclose(result_eager, result_jit, atol=1e-10)


def test_check_default_output_is_finite():
    fitted = _make_fitted()
    gof = glmax.check(fitted)
    for field in (gof.deviance, gof.aic, gof.bic, gof.df_resid, gof.pearson_chi2, gof.dispersion):
        assert jnp.isfinite(field)
