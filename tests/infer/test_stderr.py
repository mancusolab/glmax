# pattern: Imperative Shell

import pytest

import equinox as eqx
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax.random as jr

import glmax

from glmax import GLMData
from glmax._infer.stderr import FisherInfoError, HuberError
from glmax.family import Gaussian


def _make_gaussian_xy(n: int = 30, p: int = 2, seed: int = 0):
    key = jr.PRNGKey(seed)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.ones(p) + jr.normal(jr.PRNGKey(seed + 1), (n,)) * 0.2
    return X, y


def _with_dispersion(fitted, disp):
    params = fitted.params._replace(disp=jnp.asarray(disp))
    return eqx.tree_at(lambda tree: tree.result.params, fitted, params)


def _expected_huber_covariance(fitted):
    X = fitted.result.X
    phi = jnp.asarray(fitted.params.disp)
    w_pure = fitted.result.glm_wt * phi
    bread = phi * jla.inv((X * w_pure[:, jnp.newaxis]).T @ X)

    # `glm_wt` is the pure GLM weight; fitted `phi` supplies the covariance scale.
    score_no_x = fitted.result.glm_wt * fitted.result.score_residual / phi
    meat = (X * (score_no_x**2)[:, jnp.newaxis]).T @ X
    return bread @ meat @ bread


def test_fisher_info_error_jit_safe_and_finite():
    """FisherInfoError is JIT-safe: filter_jit output is finite."""
    X, y = _make_gaussian_xy()
    family = Gaussian()
    model = glmax.specify(family=family)
    result = glmax.fit(model, GLMData(X=X, y=y))

    estimator = FisherInfoError()

    def _jit_fisher(result):
        return estimator(result)

    cov_jit = eqx.filter_jit(_jit_fisher)(result)

    assert bool(
        jnp.all(jnp.isfinite(cov_jit))
    ), f"JIT FisherInfoError covariance must be finite; got diag={jnp.diag(cov_jit)}"


def test_fisher_info_error_scales_by_phi():
    """FisherInfoError covariance == phi * inv(X'W_pure X)."""
    n, p = 50, 3
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    true_beta = jnp.array([2.0, 1.0, -0.5])
    sigma = 0.5
    y = X @ true_beta + jr.normal(jr.PRNGKey(1), (n,)) * sigma

    model = glmax.specify(family=Gaussian())
    data = GLMData(X=X, y=y)
    result = glmax.fit(model, data)

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


def test_huber_error_finite_and_symmetric() -> None:
    X, y = _make_gaussian_xy()
    model = glmax.specify(family=Gaussian())
    result = glmax.fit(model, GLMData(X=X, y=y))

    cov = HuberError()(result)

    assert bool(jnp.all(jnp.isfinite(cov))), f"HuberError covariance must be finite; got diag={jnp.diag(cov)}"
    assert bool(jnp.allclose(cov, cov.T, atol=1e-6)), "HuberError covariance must be symmetric"


@pytest.mark.parametrize(
    ("estimator",),
    [
        pytest.param(FisherInfoError(), id="fisher"),
        pytest.param(HuberError(), id="huber"),
    ],
)
def test_stderr_estimators_reject_nonpositive_fitted_dispersion(estimator) -> None:
    X, y = _make_gaussian_xy()
    fitted = glmax.fit(glmax.specify(family=Gaussian()), GLMData(X=X, y=y))
    invalid_fitted = _with_dispersion(fitted, 0.0)

    with pytest.raises(ValueError, match="fitted.params.disp"):
        estimator(invalid_fitted)


def test_huber_error_uses_fitted_dispersion_as_phi_source_of_truth() -> None:
    X, y = _make_gaussian_xy(seed=11)
    fitted = glmax.fit(glmax.specify(family=Gaussian()), GLMData(X=X, y=y))
    scaled_fitted = _with_dispersion(fitted, jnp.asarray(fitted.params.disp) * 3.0)

    cov = HuberError()(scaled_fitted)
    expected_cov = _expected_huber_covariance(scaled_fitted)

    assert bool(jnp.allclose(cov, expected_cov, atol=1e-5)), (
        "HuberError must use fitted.params.disp as phi.\n"
        f"phi={scaled_fitted.params.disp}\nactual diag={jnp.diag(cov)}\nexpected diag={jnp.diag(expected_cov)}"
    )


def test_gaussian_se_equals_ols_formula() -> None:
    key = jr.PRNGKey(7)
    n, p = 100, 3
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    true_beta = jnp.array([2.0, 1.0, -0.5])
    sigma = 0.5
    y = X @ true_beta + jr.normal(jr.PRNGKey(8), (n,)) * sigma

    model = glmax.specify(family=Gaussian())
    result = glmax.fit(model, GLMData(X=X, y=y))
    inference = glmax.infer(result)

    phi = result.params.disp
    w_pure = result.glm_wt * phi
    information = (result.X * w_pure[:, jnp.newaxis]).T @ result.X
    expected_cov = phi * jla.inv(information)
    expected_se = jnp.sqrt(jnp.diag(expected_cov))

    assert jnp.allclose(
        inference.se, expected_se, atol=1e-5
    ), f"SE mismatch: got {inference.se}, expected {expected_se}"


def test_gaussian_se_positive_and_finite() -> None:
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


def test_gaussian_dispersion_positive_after_irls() -> None:
    n, p = 50, 2
    key = jr.PRNGKey(0)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.ones(p) + jr.normal(jr.PRNGKey(1), (n,)) * 0.1

    result = glmax.fit(glmax.specify(), GLMData(X=X, y=y))

    assert float(result.params.disp) > 0, f"Gaussian disp must be > 0, got {result.params.disp}"
