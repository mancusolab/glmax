# pattern: Imperative Shell
"""TDD tests for Phase 5 tasks 1–6.

Covers:
- Task 1: standalone wald_test importable from glmax._infer.inference
- Task 2: _infer() uses standalone wald_test (not model.wald_test)
- Task 3: FisherInfoError scales covariance by phi
- Task 4: IRLSFitter in fit.py
- Task 5/6 (AC3 spot checks): GLM stripped to pure noun
"""

import jax.numpy as jnp
import jax.random as jr

import glmax

from glmax import GLMData
from glmax.family import Gaussian, Poisson


# ---------------------------------------------------------------------------
# Task 1: standalone wald_test
# ---------------------------------------------------------------------------


def test_standalone_wald_test_is_importable():
    """expfam-port.AC3.5: from glmax._infer.inference import wald_test succeeds."""
    from glmax._infer.infer import wald_test  # noqa: F401

    assert callable(wald_test)


def test_standalone_wald_test_is_in_all():
    """wald_test is exported from glmax._infer.inference.__all__."""
    import importlib

    infer_module = importlib.import_module("glmax._infer.infer")
    assert "wald_test" in infer_module.__all__


def test_standalone_wald_test_gaussian_uses_t_distribution():
    """Gaussian wald_test produces p-values in (0, 1] using t-distribution."""
    from glmax._infer.infer import wald_test

    statistic = jnp.array([2.0, -1.5, 0.0])
    p = wald_test(statistic, df=50, family=Gaussian())

    assert p.shape == (3,)
    assert bool(jnp.all(p > 0))
    assert bool(jnp.all(p <= 1.0))
    # z=0 → p=1
    assert bool(jnp.isclose(p[2], 1.0, atol=1e-5))


def test_standalone_wald_test_non_gaussian_uses_normal():
    """Non-Gaussian wald_test produces p-values consistent with normal distribution."""
    from glmax._infer.infer import wald_test

    statistic = jnp.array([1.96])
    p_poisson = wald_test(statistic, df=100, family=Poisson())
    # 2 * norm.sf(1.96) ≈ 0.05
    assert bool(jnp.abs(p_poisson[0] - 0.05) < 0.005)


# ---------------------------------------------------------------------------
# Task 2: _infer() uses standalone wald_test (not model.wald_test)
# ---------------------------------------------------------------------------


def _make_gaussian_fit():
    n, p = 30, 2
    key = jr.PRNGKey(42)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.array([1.0, 0.5]) + jr.normal(jr.PRNGKey(1), (n,)) * 0.1
    model = glmax.specify(family=Gaussian())
    data = GLMData(X=X, y=y)
    result = glmax.fit(model, data)
    return result


def test_infer_does_not_call_model_wald_test(monkeypatch):
    """_infer() must not call model.wald_test after Task 2."""
    from glmax.glm import GLM

    fitted = _make_gaussian_fit()

    def _reject(*_args, **_kwargs):
        raise AssertionError("model.wald_test should not be called by _infer()")

    monkeypatch.setattr(GLM, "wald_test", _reject, raising=False)

    # Should not raise
    ir = glmax.infer(fitted)
    assert bool(jnp.all(jnp.isfinite(ir.p)))


# ---------------------------------------------------------------------------
# Task 3: FisherInfoError scales by phi
# ---------------------------------------------------------------------------


def test_fisher_info_error_scales_by_phi():
    """FisherInfoError covariance == phi * inv(X'W_pure X) where W_pure = weight * phi.

    This renormalization ensures correct SEs for families (like Gaussian) where
    IRLS weights already encode phi in the variance function.
    """
    from glmax._infer.stderr import FisherInfoError

    n, p = 50, 3
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    true_beta = jnp.array([2.0, 1.0, -0.5])
    sigma = 0.5
    y = X @ true_beta + jr.normal(jr.PRNGKey(1), (n,)) * sigma

    model = glmax.specify(family=Gaussian())
    data = GLMData(X=X, y=y)
    result = glmax.fit(model, data)

    # Expected: phi * inv(X' W_pure X), reconstructed from fit artifacts.
    phi = result.params.disp
    w_pure = result.glm_wt * phi
    infor = (result.X * w_pure[:, jnp.newaxis]).T @ result.X
    expected_cov = phi * jnp.linalg.inv(infor)

    estimator = FisherInfoError()
    actual_cov = estimator(result)

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
    fitted = glmax.fit(model, GLMData(X=X, y=y))
    result = glmax.infer(fitted)

    assert bool(jnp.all(result.se > 0)), "SEs must be positive"
    assert bool(jnp.all(jnp.isfinite(result.se))), "SEs must be finite"


# ---------------------------------------------------------------------------
# Task 4: IRLSFitter in fit.py
# ---------------------------------------------------------------------------


def test_irls_fitter_importable_from_fit():
    """IRLSFitter is importable from glmax._fit."""
    from glmax._fit import IRLSFitter  # noqa: F401

    assert callable(IRLSFitter)


def test_fit_signature_uses_irls_fitter_default():
    """fit(...) uses an IRLSFitter default directly in the signature."""
    import inspect

    from glmax._fit import fit, IRLSFitter

    default = inspect.signature(fit).parameters["fitter"].default
    assert isinstance(default, IRLSFitter)


def test_irls_fitter_returns_fit_artifacts_without_inference_summaries():
    """IRLSFitter produces fit artifacts and leaves summaries to _infer()."""
    from glmax._fit import IRLSFitter

    n, p = 50, 3
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.ones(p) + jr.normal(jr.PRNGKey(1), (n,)) * 0.1

    model = glmax.specify(family=Gaussian())
    fitter = IRLSFitter()
    result = fitter(model, GLMData(X=X, y=y))

    current_fit_result_type = __import__("glmax._fit", fromlist=["FitResult"]).FitResult
    assert isinstance(result, current_fit_result_type)
    assert bool(jnp.all(jnp.isfinite(result.params.beta)))
    assert not hasattr(result, "se")
    assert not hasattr(result, "z")
    assert not hasattr(result, "p")


def test_gaussian_dispersion_positive_after_irls():
    """expfam-port.AC2.5: Gaussian disp > 0 after fit."""
    n, p = 50, 2
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.ones(p) + jr.normal(jr.PRNGKey(1), (n,)) * 0.1

    result = glmax.fit(glmax.specify(), GLMData(X=X, y=y))

    assert float(result.params.disp) > 0, f"Gaussian disp must be > 0, got {result.params.disp}"
