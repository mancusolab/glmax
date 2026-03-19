import pytest
import statsmodels.api as sm

from conftest import fit_binomial, fit_gamma, fit_gaussian, fit_negative_binomial, fit_poisson

import equinox as eqx
import jax.numpy as jnp

import glmax
import glmax.diagnostics

from glmax import FitResult, FittedGLM, GLM, Params
from glmax.diagnostics import AbstractDiagnostic
from glmax.family import Binomial, Poisson


class _ConcreteOK(AbstractDiagnostic[jnp.ndarray], strict=True):
    def diagnose(self, fitted):
        return fitted.y


def _make_fitted(family, y, mu, disp=1.0, aux=None):
    X = jnp.ones((y.shape[0], 1))
    beta = jnp.array([0.0])
    params = Params(beta=beta, disp=jnp.asarray(disp), aux=None if aux is None else jnp.asarray(aux))
    model = GLM(family=family)
    result = FitResult(
        params=params,
        X=X,
        y=jnp.asarray(y),
        eta=jnp.zeros_like(y),
        mu=jnp.asarray(mu),
        glm_wt=jnp.ones_like(y),
        converged=jnp.asarray(True),
        num_iters=jnp.asarray(1),
        objective=jnp.asarray(0.0),
        objective_delta=jnp.asarray(0.0),
        score_residual=jnp.zeros_like(y),
    )
    return FittedGLM(model=model, result=result)


def test_abstract_diagnostic_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AbstractDiagnostic()


def test_concrete_subclass_can_be_instantiated():
    d = _ConcreteOK()
    assert isinstance(d, AbstractDiagnostic)


class TestPearsonResidual:
    def test_pearson_shape_n(self):
        fitted, _, _ = fit_gaussian()
        result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert result.shape == (5,)

    def test_pearson_gaussian_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_gaussian()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Gaussian()).fit()
        expected = sm_result.resid_pearson
        result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_pearson_poisson_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_poisson()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Poisson()).fit()
        expected = sm_result.resid_pearson
        result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_pearson_binomial_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_binomial()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Binomial()).fit()
        expected = sm_result.resid_pearson
        result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_pearson_finite_under_jit_with_boundary_means(self):
        fitted = _make_fitted(
            Poisson(),
            y=jnp.array([0.0, 1.0, 2.0]),
            mu=jnp.array([1e-300, 1e-200, 1e-100]),
        )
        result = eqx.filter_jit(glmax.diagnostics.PearsonResidual().diagnose)(fitted)
        assert jnp.all(jnp.isfinite(result))


class TestDevianceResidual:
    def test_deviance_shape_n(self):
        fitted, _, _ = fit_gaussian()
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert result.shape == (5,)

    def test_deviance_gaussian_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_gaussian()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Gaussian()).fit()
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(sm_result.resid_deviance), atol=1e-5)

    def test_deviance_poisson_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_poisson()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Poisson()).fit()
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(sm_result.resid_deviance), atol=1e-5)

    def test_deviance_binomial_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_binomial()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Binomial()).fit()
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(sm_result.resid_deviance), atol=1e-5)

    def test_deviance_poisson_zero_y_finite(self):
        fitted = _make_fitted(
            Poisson(),
            y=jnp.array([0.0, 1.0, 2.0]),
            mu=jnp.array([0.5, 1.5, 2.5]),
        )
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_deviance_all_finite(self):
        for fit_fn in [fit_gaussian, fit_poisson, fit_binomial]:
            fitted, _, _ = fit_fn()
            result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
            assert jnp.all(jnp.isfinite(result))


class TestQuantileResidual:
    def test_quantile_shape_n(self):
        fitted, _, _ = fit_gaussian()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert result.shape == (5,)

    def test_quantile_gaussian_all_finite(self):
        fitted, _, _ = fit_gaussian()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_gaussian_equals_standardised_pearson(self):
        fitted, _, _ = fit_gaussian()
        q = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        p = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert jnp.allclose(q, p, atol=1e-5)

    def test_quantile_poisson_all_finite(self):
        fitted, _, _ = fit_poisson()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_binomial_all_finite(self):
        fitted, _, _ = fit_binomial()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_gamma_all_finite(self):
        fitted, _, _ = fit_gamma()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_negative_binomial_all_finite(self):
        fitted, _, _ = fit_negative_binomial()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_deterministic(self):
        fitted, _, _ = fit_gaussian()
        r1 = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        r2 = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.allclose(r1, r2, atol=0.0)

    def test_quantile_poisson_upper_boundary_finite(self):
        fitted = _make_fitted(
            Poisson(),
            y=jnp.array([1000.0, 1000.0, 1000.0]),
            mu=jnp.array([1e-300, 1e-200, 1e-100]),
        )
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_binomial_boundary_finite(self):
        fitted = _make_fitted(
            Binomial(),
            y=jnp.array([0.0, 1.0, 0.0]),
            mu=jnp.array([1e-300, 1.0 - 1e-300, 1e-300]),
        )
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))
