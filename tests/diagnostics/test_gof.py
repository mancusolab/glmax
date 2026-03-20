import statsmodels.api as sm

from conftest import fit_gaussian, fit_poisson

import jax.numpy as jnp

from glmax.diagnostics import GofStats, GoodnessOfFit


class TestGofStats:
    def test_gof_stats_is_eqx_module(self):
        fitted, _, _ = fit_gaussian()
        result = GoodnessOfFit().diagnose(fitted)
        assert isinstance(result, GofStats)

    def test_gof_deviance_gaussian_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_gaussian()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Gaussian()).fit()
        gof = GoodnessOfFit().diagnose(fitted)
        assert jnp.allclose(gof.deviance, sm_result.deviance, atol=1e-5)

    def test_gof_deviance_poisson_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_poisson()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Poisson()).fit()
        gof = GoodnessOfFit().diagnose(fitted)
        assert jnp.allclose(gof.deviance, sm_result.deviance, atol=1e-5)

    def test_gof_aic_gaussian_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_gaussian()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Gaussian()).fit()
        gof = GoodnessOfFit().diagnose(fitted)
        assert jnp.allclose(gof.aic, sm_result.aic, atol=1e-4)

    def test_gof_bic_gaussian_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_gaussian()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Gaussian()).fit()
        gof = GoodnessOfFit().diagnose(fitted)
        assert jnp.allclose(gof.bic, sm_result.bic_llf, atol=1e-4)

    def test_gof_df_resid_is_n_minus_p(self):
        fitted, X_raw, y_raw = fit_gaussian()
        n, p = X_raw.shape
        gof = GoodnessOfFit().diagnose(fitted)
        assert jnp.allclose(gof.df_resid, n - p, atol=0.0)

    def test_gof_pearson_chi2_equals_formula(self):
        fitted, _, _ = fit_gaussian()
        family = fitted.family
        y, mu = fitted.y, fitted.mu
        disp, aux = fitted.params.disp, fitted.params.aux
        v = family.variance(mu, disp, aux=aux)
        expected = jnp.sum((y - mu) ** 2 / v)
        gof = GoodnessOfFit().diagnose(fitted)
        assert jnp.allclose(gof.pearson_chi2, expected, atol=1e-10)

    def test_gof_all_fields_finite(self):
        for fit_fn in [fit_gaussian, fit_poisson]:
            fitted, _, _ = fit_fn()
            gof = GoodnessOfFit().diagnose(fitted)
            for field in (
                gof.deviance,
                gof.pearson_chi2,
                gof.df_resid,
                gof.dispersion,
                gof.aic,
                gof.bic,
            ):
                assert jnp.isfinite(field)
