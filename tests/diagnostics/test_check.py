# pattern: Imperative Shell

import pytest

import equinox as eqx
import jax.numpy as jnp

import glmax

from glmax import GLMData
from glmax.diagnostics import (
    DEFAULT_DIAGNOSTICS,
    DevianceResidual,
    GofStats,
    InfluenceStats,
    PearsonResidual,
)
from glmax.family import Gaussian


def _make_fitted():
    model = glmax.specify(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        y=jnp.array([1.2, 1.9, 3.1, 4.2]),
    )
    return glmax.fit(model, data)


def test_check_default_returns_5_tuple():
    fitted = _make_fitted()
    result = glmax.check(fitted)
    assert isinstance(result, tuple)
    assert len(result) == 5
    assert len(DEFAULT_DIAGNOSTICS) == 5


def test_check_default_positional_types():
    fitted = _make_fitted()
    pearson, deviance, quantile, gof, influence = glmax.check(fitted)
    assert isinstance(pearson, jnp.ndarray)
    assert isinstance(deviance, jnp.ndarray)
    assert isinstance(quantile, jnp.ndarray)
    assert isinstance(gof, GofStats)
    assert isinstance(influence, InfluenceStats)


def test_check_custom_diagnostics_single():
    fitted = _make_fitted()
    result = glmax.check(fitted, diagnostics=(PearsonResidual(),))
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (4,)


def test_check_custom_diagnostics_two():
    fitted = _make_fitted()
    result = glmax.check(fitted, diagnostics=(PearsonResidual(), DevianceResidual()))
    assert len(result) == 2


def test_check_rejects_non_fitted_glm():
    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.check("not_a_fitted_glm")


def test_check_filter_jit_produces_same_result():
    fitted = _make_fitted()
    result_eager = glmax.check(fitted, diagnostics=(PearsonResidual(),))
    result_jit = eqx.filter_jit(glmax.check)(fitted, diagnostics=(PearsonResidual(),))
    assert jnp.allclose(result_eager[0], result_jit[0], atol=1e-10)


def test_check_default_all_outputs_finite():
    fitted = _make_fitted()
    pearson, deviance, quantile, gof, influence = glmax.check(fitted)
    assert jnp.all(jnp.isfinite(pearson))
    assert jnp.all(jnp.isfinite(deviance))
    assert jnp.all(jnp.isfinite(quantile))
    for field in (gof.deviance, gof.aic, gof.bic, gof.df_resid, gof.pearson_chi2, gof.dispersion):
        assert jnp.isfinite(field)
    assert jnp.all(jnp.isfinite(influence.leverage))
    assert jnp.all(jnp.isfinite(influence.cooks_distance))
