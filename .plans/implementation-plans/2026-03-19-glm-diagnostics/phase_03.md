# GLM Diagnostics Implementation Plan — Phase 3

**Goal:** Add `GofStats`, `GoodnessOfFit`, `InfluenceStats`, and `Influence` to `src/glmax/diagnostics.py`.

**Architecture:** Single file continuation. All four additions go into the existing `diagnostics.py`. `GofStats` and `InfluenceStats` are typed result containers (`eqx.Module, strict=True`). `GoodnessOfFit` and `Influence` are concrete `AbstractDiagnostic` implementations. `Influence` recomputes the Cholesky factor of `X^T W X` using `jax.scipy.linalg.cholesky` + `solve_triangular`.

**Tech Stack:** JAX, Equinox, `jax.scipy.linalg`, `jax.numpy`

**Scope:** Phase 3 of 4 — depends on Phase 1 (`deviance_contribs`) and Phase 2 (`AbstractDiagnostic`, `PearsonResidual`)

**Codebase verified:** 2026-03-19

---

## Acceptance Criteria Coverage

### glm-diagnostics.AC5: GoodnessOfFit / GofStats
- **glm-diagnostics.AC5.1 Success:** `GofStats.deviance` matches statsmodels `deviance` for Gaussian and Poisson
- **glm-diagnostics.AC5.2 Success:** `GofStats.aic` matches statsmodels `aic`; `GofStats.bic` matches statsmodels `bic`
- **glm-diagnostics.AC5.3 Success:** `GofStats.df_resid` equals `n - p`
- **glm-diagnostics.AC5.4 Success:** `GofStats.pearson_chi2` equals `sum((y - mu)² / V(mu))`

### glm-diagnostics.AC6: Influence / InfluenceStats
- **glm-diagnostics.AC6.1 Success:** `InfluenceStats.leverage` matches statsmodels `get_influence().hat_matrix_diag` for Gaussian and Poisson to float64 tolerance
- **glm-diagnostics.AC6.2 Success:** All leverage values satisfy `0 < h_i < 1`
- **glm-diagnostics.AC6.3 Success:** `InfluenceStats.cooks_distance` is non-negative for all observations
- **glm-diagnostics.AC6.4 Success:** Cook's distance equals `pearson_r_i² * h_i / (p * (1 - h_i)²)`

---

## Key Codebase Facts

- `fitted.X` shape `(n, p)`, `fitted.y` shape `(n,)`, `fitted.mu` shape `(n,)`, `fitted.eta` shape `(n,)`, `fitted.glm_wt` shape `(n,)` = IRLS weights `w_i = 1/(V(mu_i)*g'(mu_i)^2)`
- `fitted.params.disp` = converged dispersion; `fitted.params.aux` = family-specific aux
- `fitted.model.log_prob(y, eta, disp, aux)` = SCALAR total log-likelihood (sum over observations)
- `n, p = fitted.X.shape`
- AIC/BIC: no existing implementation in codebase — novel additions
- Leverage: `jax.scipy.linalg.cholesky(a, lower=True)` and `jax.scipy.linalg.solve_triangular(a, b, lower=True)` — verified available
- Weight broadcast pattern from `_fit/solve.py`: `X * w[:, jnp.newaxis]`
- Cook's D: `D_i = r_i^2 * h_i / (p * (1 - h_i)^2)` where `r_i` is Pearson residual and `p = X.shape[1]`
- Test file: add to `tests/diagnostics/test_gof.py` and `tests/diagnostics/test_influence.py`
- Run tests: `pytest -p no:capture tests`

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->

<!-- START_TASK_1 -->
### Task 1: Add `GofStats` and `GoodnessOfFit`

**Verifies:** glm-diagnostics.AC5.1, glm-diagnostics.AC5.2, glm-diagnostics.AC5.3, glm-diagnostics.AC5.4

**Files:**
- Modify: `src/glmax/diagnostics.py` (add after `QuantileResidual`)
- Modify: `src/glmax/diagnostics.py` `__all__`
- Test: `tests/diagnostics/test_gof.py` (new file)

**Implementation:**

Add to `diagnostics.py` after `QuantileResidual`:

```python
class GofStats(eqx.Module, strict=True):
    r"""Goodness-of-fit statistics for a fitted GLM.

    All fields are scalar JAX arrays. Pytree-compatible.

    **Fields:**

    - `deviance`: total deviance $D = \sum_i d_i$.
    - `pearson_chi2`: Pearson chi-squared $\sum_i (y_i - \mu_i)^2 / V(\mu_i)$.
    - `df_resid`: residual degrees of freedom $n - p$.
    - `dispersion`: fitted dispersion parameter $\hat\phi$.
    - `aic`: Akaike information criterion $-2\ell + 2p$.
    - `bic`: Bayesian information criterion $-2\ell + p \log n$.
    """

    deviance: Array
    pearson_chi2: Array
    df_resid: Array
    dispersion: Array
    aic: Array
    bic: Array


class GoodnessOfFit(AbstractDiagnostic[GofStats], strict=True):
    r"""Goodness-of-fit statistics: deviance, Pearson chi-squared, AIC, BIC, dispersion."""

    def diagnose(self, fitted: FittedGLM) -> GofStats:
        r"""Compute goodness-of-fit statistics.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

        **Returns:**

        `GofStats` with scalar array fields.
        """
        family = fitted.model.family
        y = fitted.y
        mu = fitted.mu
        eta = fitted.eta
        disp = fitted.params.disp
        aux = fitted.params.aux
        n, p = fitted.X.shape

        # Deviance
        d_contribs = family.deviance_contribs(y, mu, disp, aux=aux)
        deviance = jnp.sum(d_contribs)

        # Pearson chi-squared
        v = family.variance(mu, disp, aux=aux)
        pearson_chi2 = jnp.sum((y - mu) ** 2 / v)

        # Degrees of freedom
        df_resid = jnp.asarray(n - p, dtype=jnp.float64)

        # Log-likelihood
        ll = fitted.model.log_prob(y, eta, disp, aux=aux)
        n_f = jnp.asarray(n, dtype=jnp.float64)
        p_f = jnp.asarray(p, dtype=jnp.float64)

        # AIC and BIC
        aic = -2.0 * ll + 2.0 * p_f
        bic = -2.0 * ll + p_f * jnp.log(n_f)

        return GofStats(
            deviance=deviance,
            pearson_chi2=pearson_chi2,
            df_resid=df_resid,
            dispersion=jnp.asarray(disp),
            aic=aic,
            bic=bic,
        )
```

Add `"GofStats"`, `"GoodnessOfFit"` to `__all__`.

**Testing** (new file `tests/diagnostics/test_gof.py`):

```python
# pattern: Imperative Shell

import numpy as np
import pytest
import statsmodels.api as sm

import jax.numpy as jnp

import glmax
from glmax import GLMData
from glmax.diagnostics import GoodnessOfFit, GofStats

from .conftest import fit_gaussian, fit_poisson


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
        # bic_llf = -2*ll + p*log(n); statsmodels sm_result.bic is bic_deviance — use bic_llf
        assert jnp.allclose(gof.bic, sm_result.bic_llf, atol=1e-4)

    def test_gof_df_resid_is_n_minus_p(self):
        fitted, X_raw, y_raw = fit_gaussian()
        n, p = X_raw.shape
        gof = GoodnessOfFit().diagnose(fitted)
        assert jnp.allclose(gof.df_resid, n - p, atol=0.0)

    def test_gof_pearson_chi2_equals_formula(self):
        fitted, _, _ = fit_gaussian()
        family = fitted.model.family
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
            for field in (gof.deviance, gof.pearson_chi2, gof.df_resid, gof.dispersion, gof.aic, gof.bic):
                assert jnp.isfinite(field)
```

**Verification:**

Run: `pytest -p no:capture tests/diagnostics/test_gof.py`
Expected: All tests pass

Run: `pytest -p no:capture tests`
Expected: All tests pass

**Commit:** `feat(diagnostics): add GofStats and GoodnessOfFit`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add `InfluenceStats` and `Influence`

**Verifies:** glm-diagnostics.AC6.1, glm-diagnostics.AC6.2, glm-diagnostics.AC6.3, glm-diagnostics.AC6.4

**Files:**
- Modify: `src/glmax/diagnostics.py` (add after `GoodnessOfFit`)
- Modify: `src/glmax/diagnostics.py` `__all__`
- Add import: `from jax.scipy import linalg as jscla` at module top of `diagnostics.py`
- Test: `tests/diagnostics/test_influence.py` (new file)

**Implementation:**

Add at the top of `diagnostics.py` (alongside existing imports):
```python
from jax.scipy import linalg as jscla
```

Add after `GoodnessOfFit`:

```python
class InfluenceStats(eqx.Module, strict=True):
    r"""Per-observation influence statistics.

    **Fields:**

    - `leverage`: hat-matrix diagonal $h_{ii} \in (0, 1)$, shape `(n,)`.
    - `cooks_distance`: Cook's distance $D_i \geq 0$, shape `(n,)`.
    """

    leverage: Array
    cooks_distance: Array


class Influence(AbstractDiagnostic[InfluenceStats], strict=True):
    r"""Leverage and Cook's distance via Cholesky-based hat-matrix computation.

    Recomputes $\operatorname{chol}(X^T W X)$ from the fitted weights;
    does not rely on the Cholesky factor from IRLS (which is not persisted).
    """

    def diagnose(self, fitted: FittedGLM) -> InfluenceStats:
        r"""Compute leverage and Cook's distance.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

        **Returns:**

        `InfluenceStats` with `leverage` and `cooks_distance`, each shape `(n,)`.
        """
        X = fitted.X
        y = fitted.y
        mu = fitted.mu
        w = fitted.glm_wt          # shape (n,), w_i = 1 / (V(mu_i) * g'(mu_i)^2)
        disp = fitted.params.disp
        aux = fitted.params.aux
        _, p = X.shape

        # Leverage: h_i = diag(W^{1/2} X (X^T W X)^{-1} X^T W^{1/2})
        # via Z = L^{-1} (W^{1/2} X)^T, L = chol(X^T W X), h_i = sum(Z[:, i]^2)
        sqrt_w = jnp.sqrt(w)                          # (n,)
        Xw = X * sqrt_w[:, jnp.newaxis]               # (n, p)  W^{1/2} X
        A = Xw.T @ Xw                                 # (p, p)  X^T W X
        L = jscla.cholesky(A, lower=True)             # (p, p)  lower-triangular
        Z = jscla.solve_triangular(L, Xw.T, lower=True)  # (p, n)  L^{-1} (W^{1/2} X)^T
        h = jnp.sum(Z ** 2, axis=0)                   # (n,)

        # Pearson residuals for Cook's distance
        v = fitted.model.family.variance(mu, disp, aux=aux)
        r_pearson = (y - mu) / jnp.sqrt(v)            # (n,)

        # Cook's distance: D_i = r_i^2 * h_i / (p * (1 - h_i)^2)
        p_f = jnp.asarray(p, dtype=jnp.float64)
        cooks_d = r_pearson ** 2 * h / (p_f * (1.0 - h) ** 2)

        return InfluenceStats(leverage=h, cooks_distance=cooks_d)
```

Add `"InfluenceStats"`, `"Influence"` to `__all__`.

**Testing** (new file `tests/diagnostics/test_influence.py`):

```python
# pattern: Imperative Shell

import numpy as np
import pytest
import statsmodels.api as sm

import jax.numpy as jnp

import glmax
from glmax import GLMData
from glmax.diagnostics import Influence, InfluenceStats

from .conftest import fit_gaussian, fit_poisson


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
        expected = r ** 2 * h / (p * (1 - h) ** 2)
        assert jnp.allclose(result.cooks_distance, expected, atol=1e-10)

    def test_leverage_sums_to_p(self):
        # trace(H) = p for GLMs
        fitted, X_raw, _ = fit_gaussian()
        result = Influence().diagnose(fitted)
        _, p = X_raw.shape
        assert jnp.allclose(jnp.sum(result.leverage), p, atol=1e-8)
```

**Verification:**

Run: `pytest -p no:capture tests/diagnostics/test_influence.py`
Expected: All tests pass

Run: `pytest -p no:capture tests`
Expected: All tests pass

**Commit:** `feat(diagnostics): add InfluenceStats and Influence`
<!-- END_TASK_2 -->

<!-- END_SUBCOMPONENT_A -->
