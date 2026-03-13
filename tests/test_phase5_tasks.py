# pattern: Imperative Shell
"""TDD tests for Phase 5 tasks 1–6.

Covers:
- Task 1: standalone wald_test importable from glmax.infer.inference
- Task 2: infer() uses standalone wald_test (not model.wald_test)
- Task 3: FisherInfoError scales covariance by phi
- Task 4: IRLSFitter in fit.py; DEFAULT_FITTER = IRLSFitter()
- Task 5/6 (AC3 spot checks): GLM stripped to pure noun
"""

import jax.numpy as jnp
import jax.random as jr

import glmax

from glmax import FitResult, GLMData
from glmax.family import Gaussian, Poisson


# ---------------------------------------------------------------------------
# Task 1: standalone wald_test
# ---------------------------------------------------------------------------


def test_standalone_wald_test_is_importable():
    """expfam-port.AC3.5: from glmax.infer.inference import wald_test succeeds."""
    from glmax.infer.inference import wald_test  # noqa: F401

    assert callable(wald_test)


def test_standalone_wald_test_is_in_all():
    """wald_test is exported from glmax.infer.inference.__all__."""
    from glmax.infer import inference

    assert "wald_test" in inference.__all__


def test_standalone_wald_test_gaussian_uses_t_distribution():
    """Gaussian wald_test produces p-values in (0, 1] using t-distribution."""
    from glmax.infer.inference import wald_test

    statistic = jnp.array([2.0, -1.5, 0.0])
    p = wald_test(statistic, df=50, family=Gaussian())

    assert p.shape == (3,)
    assert bool(jnp.all(p > 0))
    assert bool(jnp.all(p <= 1.0))
    # z=0 → p=1
    assert bool(jnp.isclose(p[2], 1.0, atol=1e-5))


def test_standalone_wald_test_non_gaussian_uses_normal():
    """Non-Gaussian wald_test produces p-values consistent with normal distribution."""
    from glmax.infer.inference import wald_test

    statistic = jnp.array([1.96])
    p_poisson = wald_test(statistic, df=100, family=Poisson())
    # 2 * norm.sf(1.96) ≈ 0.05
    assert bool(jnp.abs(p_poisson[0] - 0.05) < 0.005)


# ---------------------------------------------------------------------------
# Task 2: infer() uses standalone wald_test (not model.wald_test)
# ---------------------------------------------------------------------------


def _make_gaussian_fit():
    n, p = 30, 2
    key = jr.PRNGKey(42)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.array([1.0, 0.5]) + jr.normal(jr.PRNGKey(1), (n,)) * 0.1
    model = glmax.specify(family=Gaussian())
    data = GLMData(X=X, y=y)
    result = glmax.fit(model, data)
    return model, result


def test_infer_does_not_call_model_wald_test(monkeypatch):
    """infer() must not call model.wald_test after Task 2."""
    from glmax.glm import GLM

    model, fit_result = _make_gaussian_fit()

    def _reject(*_args, **_kwargs):
        raise AssertionError("model.wald_test should not be called by infer()")

    monkeypatch.setattr(GLM, "wald_test", _reject)

    # Should not raise
    ir = glmax.infer(model, fit_result)
    assert bool(jnp.all(jnp.isfinite(ir.p)))


# ---------------------------------------------------------------------------
# Task 3: FisherInfoError scales by phi
# ---------------------------------------------------------------------------


def test_fisher_info_error_scales_by_phi():
    """FisherInfoError covariance == phi * inv(X'W_pure X) where W_pure = weight * phi.

    This renormalization ensures correct SEs for families (like Gaussian) where
    IRLS weights already encode phi in the variance function.
    """
    from glmax.infer.stderr import FisherInfoError

    n, p = 50, 3
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    true_beta = jnp.array([2.0, 1.0, -0.5])
    sigma = 0.5
    y = X @ true_beta + jr.normal(jr.PRNGKey(1), (n,)) * sigma

    model = glmax.specify(family=Gaussian())
    data = GLMData(X=X, y=y)
    result = glmax.fit(model, data)

    eta = result.eta
    mu = result.mu
    weight = result.glm_wt

    # Expected: phi * inv(X' W_pure X) where W_pure = weight * phi
    phi = model.family.scale(X, y, mu)
    w_pure = weight * phi
    infor = (X * w_pure[:, jnp.newaxis]).T @ X
    expected_cov = phi * jnp.linalg.inv(infor)

    estimator = FisherInfoError()
    actual_cov = estimator(model.family, X, y, eta, mu, weight, result.params.disp)

    assert bool(jnp.allclose(actual_cov, expected_cov, atol=1e-5)), (
        f"FisherInfoError does not scale by phi correctly.\n"
        f"phi={phi}\nactual diag={jnp.diag(actual_cov)}\nexpected diag={jnp.diag(expected_cov)}"
    )


def test_gaussian_se_positive_and_finite():
    """expfam-port.AC3.6 (partial): Gaussian SEs are positive and finite."""
    n, p = 100, 3
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    true_beta = jnp.array([2.0, 1.0, -0.5])
    y = X @ true_beta + jr.normal(jr.PRNGKey(1), (n,)) * 0.5

    model = glmax.specify(family=Gaussian())
    result = glmax.fit(model, GLMData(X=X, y=y))

    assert bool(jnp.all(result.se > 0)), "SEs must be positive"
    assert bool(jnp.all(jnp.isfinite(result.se))), "SEs must be finite"


# ---------------------------------------------------------------------------
# Task 4: IRLSFitter in fit.py
# ---------------------------------------------------------------------------


def test_irls_fitter_importable_from_fit():
    """IRLSFitter is importable from glmax.fit."""
    from glmax.fit import IRLSFitter  # noqa: F401

    assert callable(IRLSFitter)


def test_default_fitter_is_irls_fitter():
    """DEFAULT_FITTER is an IRLSFitter instance."""
    from glmax.fit import DEFAULT_FITTER, IRLSFitter

    assert isinstance(DEFAULT_FITTER, IRLSFitter)


def test_irls_fitter_returns_finite_fitresult():
    """IRLSFitter produces a FitResult with all fields finite (Gaussian)."""
    from glmax.fit import IRLSFitter

    n, p = 50, 3
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.ones(p) + jr.normal(jr.PRNGKey(1), (n,)) * 0.1

    model = glmax.specify(family=Gaussian())
    fitter = IRLSFitter()
    result = fitter(model, GLMData(X=X, y=y))

    assert isinstance(result, FitResult)
    assert bool(jnp.all(jnp.isfinite(result.params.beta)))
    assert bool(jnp.all(jnp.isfinite(result.se)))
    assert bool(jnp.all(jnp.isfinite(result.p)))


def test_gaussian_dispersion_positive_after_irls():
    """expfam-port.AC2.5: Gaussian disp > 0 after fit."""
    n, p = 50, 2
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.ones(p) + jr.normal(jr.PRNGKey(1), (n,)) * 0.1

    result = glmax.fit(glmax.specify(), GLMData(X=X, y=y))

    assert float(result.params.disp) > 0, f"Gaussian disp must be > 0, got {result.params.disp}"
