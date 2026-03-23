# pattern: Imperative Shell

import pytest

import jax
import jax.numpy as jnp

import glmax

from glmax.family import Binomial, Gamma, Gaussian, Poisson


def test_fit_jvp_gaussian_beta_tangent_matches_analytical():
    # For Gaussian/identity, beta = (X^T X)^{-1} X^T y,
    # so dbeta/dy in direction dy = (X^T X)^{-1} X^T dy.
    family = Gaussian()
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([1.0, 2.0, 3.0, 4.0])
    dy = jnp.array([1.0, 0.0, 0.0, 0.0])
    dX = jnp.zeros_like(X)

    _, dbeta = jax.jvp(
        lambda X_, y_: glmax.fit(family, X_, y_).params.beta,
        (X, y),
        (dX, dy),
    )

    expected = jnp.linalg.solve(X.T @ X, X.T @ dy)
    assert jnp.allclose(dbeta, expected, atol=1e-4)


def test_fit_jvp_gaussian_beta_tangent_matches_finite_difference():
    family = Gaussian()
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([1.1, 1.9, 3.1, 4.0])
    dy = jnp.array([0.1, -0.1, 0.0, 0.05])
    dX = jnp.zeros_like(X)

    _, dbeta_jvp = jax.jvp(
        lambda X_, y_: glmax.fit(family, X_, y_).params.beta,
        (X, y),
        (dX, dy),
    )

    eps = 1e-4
    beta_p = glmax.fit(family, X, y + eps * dy).params.beta
    beta_m = glmax.fit(family, X, y - eps * dy).params.beta
    dbeta_fd = (beta_p - beta_m) / (2 * eps)

    assert jnp.allclose(dbeta_jvp, dbeta_fd, atol=1e-3)


@pytest.mark.parametrize(
    ("family", "X", "y"),
    [
        (
            Gaussian(),
            jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),
            jnp.array([0.1, 1.0, 2.1, 2.9, 4.2]),
        ),
        (
            Gamma(),
            jnp.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]]),
            jnp.array([0.8, 1.1, 1.7, 2.2, 2.9]),
        ),
        (
            Poisson(),
            jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),
            jnp.array([0.0, 1.0, 1.0, 2.0, 3.0]),
        ),
        (
            Binomial(),
            jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]),
            jnp.array([0.0, 0.0, 1.0, 1.0, 1.0]),
        ),
    ],
)
def test_fit_jvp_returns_finite_tangents(family, X, y):
    dy = jnp.ones_like(y) * 0.01
    dX = jnp.zeros_like(X)

    fitted, tangent = jax.jvp(
        lambda X_, y_: glmax.fit(family, X_, y_),
        (X, y),
        (dX, dy),
    )

    assert jnp.all(jnp.isfinite(tangent.result.params.beta))
    assert jnp.all(jnp.isfinite(tangent.result.eta))
    assert jnp.all(jnp.isfinite(tangent.result.mu))
    assert jnp.all(jnp.isfinite(tangent.result.objective))


def test_fit_grad_sum_beta_wrt_y_is_finite():
    # jax.grad requires a scalar output; use sum(beta) as the scalar.
    family = Gaussian()
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([1.0, 2.0, 3.0, 4.0])

    grad_fn = jax.grad(lambda y_: jnp.sum(glmax.fit(family, X, y_).params.beta))
    g = grad_fn(y)

    assert g.shape == y.shape
    assert jnp.all(jnp.isfinite(g))


def test_fit_grad_gaussian_matches_analytical():
    # grad of sum(beta) w.r.t. y is row-sum of (X^T X)^{-1} X^T.
    family = Gaussian()
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([1.0, 2.0, 3.0, 4.0])

    g = jax.grad(lambda y_: jnp.sum(glmax.fit(family, X, y_).params.beta))(y)

    # grad of sum(beta) = ones @ (X^T X)^{-1} X^T
    expected = jnp.ones(2) @ jnp.linalg.solve(X.T @ X, X.T)
    assert jnp.allclose(g, expected, atol=1e-4)
