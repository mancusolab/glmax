# pattern: Functional Core
"""Tests for the Gamma exponential family (Phase 4: expfam-port).

Covers:
- expfam-port.AC2.1: Gamma._bounds is a two-tuple with positive lower bound
- expfam-port.AC2.6: Gamma with InverseLink fits a positive-response dataset without raising
- sample shape and positivity
"""

import jax
import jax.numpy as jnp
import jax.random as jr

import glmax

from glmax import GLMData
from glmax.family import Gamma


# --- AC2.1: Gamma._bounds two-tuple ---


def test_gamma_bounds():
    g = Gamma()
    assert len(g._bounds) == 2
    assert g._bounds[0] > 0  # lower bound is positive (tiny)


def test_gamma_bounds_lower_less_than_upper():
    g = Gamma()
    lo, hi = g._bounds
    assert lo < hi


# --- AC2.6: Gamma fits a positive-response dataset without raising ---


def test_gamma_fits_without_error():
    key = jr.PRNGKey(42)
    key_X, key_y = jr.split(key)
    n, p = 50, 3
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key_X, (n, p - 1))], axis=1)
    # Positive response: sample from Gamma(shape=2, scale=1)
    y = jr.gamma(key_y, 2.0, shape=(n,))

    model = glmax.specify(family=Gamma())
    data = GLMData(X=X, y=y)
    result = glmax.fit(model, data)

    assert result is not None
    assert jnp.all(jnp.isfinite(result.params.beta))
    assert jnp.isfinite(result.params.disp)


# --- sample shape and positivity ---


def test_gamma_sample_shape():
    key = jax.random.PRNGKey(0)
    # InverseLink: eta = 1/mu; use eta = 1 so mu = 1
    eta = jnp.ones(20)
    g = Gamma()
    s = g.sample(key, eta, 1.0)
    assert s.shape == (20,)
    assert jnp.all(s > 0), "Gamma samples must be positive"


def test_gamma_default_link_is_inverse():
    from glmax.family import InverseLink

    g = Gamma()
    assert isinstance(g.glink, InverseLink)


def test_gamma_variance():
    g = Gamma()
    mu = jnp.array([1.0, 2.0, 3.0])
    disp = 0.5
    v = g.variance(mu, disp)
    expected = disp * mu**2
    assert jnp.allclose(v, expected)


def test_gamma_negloglikelihood_finite():
    g = Gamma()
    n = 20
    y = jnp.ones(n) * 2.0
    # InverseLink: eta = 1/mu; mu = 2 -> eta = 0.5
    eta = jnp.ones(n) * 0.5
    nll = g.negloglikelihood(y, eta, 1.0)
    assert jnp.isfinite(nll)


def test_gamma_canonical_dispersion_passthrough():
    g = Gamma()
    result = g.canonical_dispersion(2.5)
    assert float(result) == 2.5


def test_gamma_update_dispersion_passthrough():
    g = Gamma()
    X = jnp.zeros((5, 3))
    y = jnp.ones(5)
    eta = jnp.ones(5) * 0.5
    result = g.update_dispersion(X, y, eta, disp=1.0)
    assert float(result) == 1.0


def test_gamma_estimate_dispersion_passthrough():
    g = Gamma()
    X = jnp.zeros((5, 3))
    y = jnp.ones(5)
    eta = jnp.ones(5) * 0.5
    result = g.estimate_dispersion(X, y, eta, disp=1.0)
    assert float(result) == 1.0
