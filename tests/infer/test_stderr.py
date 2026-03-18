# pattern: Imperative Shell

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
