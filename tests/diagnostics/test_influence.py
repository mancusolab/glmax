import statsmodels.api as sm

from conftest import fit_gaussian, fit_poisson

import jax.numpy as jnp

from glmax.diagnostics import Influence, InfluenceStats


class TestInfluenceStats:
    def test_influence_returns_influence_stats(self):
        fitted, _, _ = fit_gaussian()
        result = Influence().diagnose(fitted)
        assert isinstance(result, InfluenceStats)

    def test_leverage_shape_n(self):
        fitted, X_raw, _ = fit_gaussian()
        result = Influence().diagnose(fitted)
        assert result.leverage.shape == (X_raw.shape[0],)

    def test_cooks_distance_shape_n(self):
        fitted, X_raw, _ = fit_gaussian()
        result = Influence().diagnose(fitted)
        assert result.cooks_distance.shape == (X_raw.shape[0],)

    def test_leverage_gaussian_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_gaussian()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Gaussian()).fit()
        sm_h = sm_result.get_influence().hat_matrix_diag
        result = Influence().diagnose(fitted)
        assert jnp.allclose(result.leverage, jnp.array(sm_h), atol=1e-8)

    def test_leverage_poisson_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_poisson()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Poisson()).fit()
        sm_h = sm_result.get_influence().hat_matrix_diag
        result = Influence().diagnose(fitted)
        assert jnp.allclose(result.leverage, jnp.array(sm_h), atol=1e-8)

    def test_leverage_in_open_unit_interval(self):
        for fit_fn in [fit_gaussian, fit_poisson]:
            fitted, _, _ = fit_fn()
            result = Influence().diagnose(fitted)
            assert jnp.all(result.leverage > 0)
            assert jnp.all(result.leverage < 1)

    def test_cooks_distance_nonnegative(self):
        for fit_fn in [fit_gaussian, fit_poisson]:
            fitted, _, _ = fit_fn()
            result = Influence().diagnose(fitted)
            assert jnp.all(result.cooks_distance >= 0)

    def test_cooks_distance_matches_formula(self):
        fitted, _, _ = fit_gaussian()
        family = fitted.model.family
        y, mu = fitted.y, fitted.mu
        disp, aux = fitted.params.disp, fitted.params.aux
        _, p = fitted.X.shape
        v = family.variance(mu, disp, aux=aux)
        r = (y - mu) / jnp.sqrt(v)
        result = Influence().diagnose(fitted)
        h = result.leverage
        expected = r**2 * h / (p * (1 - h) ** 2)
        assert jnp.allclose(result.cooks_distance, expected, atol=1e-10)

    def test_leverage_sums_to_p(self):
        fitted, X_raw, _ = fit_gaussian()
        result = Influence().diagnose(fitted)
        _, p = X_raw.shape
        assert jnp.allclose(jnp.sum(result.leverage), p, atol=1e-8)
