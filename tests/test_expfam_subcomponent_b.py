# pattern: Functional Core
"""Tests for Subcomponent B: JAX sample, Gaussian dispersion.

Covers:
- expfam-port.AC2.3: sample(key, eta, disp) returns JAX array, shape == eta.shape
- Task 3: sample exists and is JAX-native on all 4 families; random_gen is gone
- Task 4: Gaussian.variance(mu, disp) == disp * ones_like(mu)
         Gaussian.canonical_dispersion(disp) == jnp.asarray(disp)
         Gaussian.estimate_dispersion(X, y, eta) == RSS / (n - p)
"""
import pytest

import jax
import jax.numpy as jnp

from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


_ALL_FAMILIES = [Gaussian, Poisson, Binomial, NegativeBinomial]
_KEY = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Task 3: sample
# ---------------------------------------------------------------------------


class TestSample:
    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_sample_returns_correct_shape(self, FamilyCls):
        """AC2.3: sample(key, eta, disp) returns array with shape == eta.shape."""
        f = FamilyCls()
        eta = jnp.zeros(10)
        s = f.sample(_KEY, eta, 1.0)
        assert s.shape == (10,), f"{FamilyCls.__name__}.sample shape {s.shape} != (10,)"

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_sample_returns_jax_array(self, FamilyCls):
        """sample must return a JAX array (not numpy)."""
        f = FamilyCls()
        eta = jnp.zeros(5)
        s = f.sample(_KEY, eta, 1.0)
        assert isinstance(s, jax.Array), f"{FamilyCls.__name__}.sample must return a JAX array"

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_sample_is_finite(self, FamilyCls):
        """Samples must be finite (no NaN/Inf)."""
        f = FamilyCls()
        eta = jnp.zeros(20)
        s = f.sample(_KEY, eta, 0.5)
        assert jnp.all(jnp.isfinite(s)), f"{FamilyCls.__name__}.sample produced non-finite values"

    def test_gaussian_sample_shape_matches_eta(self):
        """Gaussian: sample shape == eta shape."""
        g = Gaussian()
        eta = jnp.ones(7)
        s = g.sample(_KEY, eta, 2.0)
        assert s.shape == eta.shape

    def test_nb_sample_gamma_poisson_mixture(self):
        """NB sample uses Gamma-Poisson mixture, must be non-negative integers."""
        nb = NegativeBinomial()
        eta = jnp.full(100, jnp.log(5.0))
        s = nb.sample(_KEY, eta, 0.5)
        assert s.shape == (100,)
        assert jnp.all(s >= 0), "NB samples must be non-negative"

    def test_no_random_gen_method_on_families(self):
        """random_gen must be removed; sample replaces it."""
        for FamilyCls in _ALL_FAMILIES:
            assert not hasattr(FamilyCls(), "random_gen"), f"{FamilyCls.__name__} still has deprecated random_gen"

    def test_different_keys_produce_different_samples(self):
        """Two distinct keys must produce different Gaussian samples."""
        g = Gaussian()
        eta = jnp.zeros(10)
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        s1 = g.sample(key1, eta, 1.0)
        s2 = g.sample(key2, eta, 1.0)
        assert not jnp.allclose(s1, s2), "Different keys must yield different samples"


# ---------------------------------------------------------------------------
# Task 4: Gaussian dispersion
# ---------------------------------------------------------------------------


class TestGaussianDispersion:
    def test_variance_includes_disp(self):
        """Gaussian.variance(mu, disp) must return disp * ones_like(mu)."""
        g = Gaussian()
        mu = jnp.ones(5)
        v = g.variance(mu, 2.0)
        assert jnp.allclose(v, 2.0 * jnp.ones(5)), f"variance expected 2.0*ones, got {v}"

    def test_variance_default_disp_is_one(self):
        """Gaussian.variance(mu) defaults to disp=1.0 → ones_like(mu)."""
        g = Gaussian()
        mu = jnp.ones(4)
        v = g.variance(mu)
        assert jnp.allclose(v, jnp.ones(4)), f"variance default (disp=1) expected ones, got {v}"

    def test_canonical_dispersion_passthrough(self):
        """Gaussian.canonical_dispersion(disp) must return jnp.asarray(disp)."""
        g = Gaussian()
        cd = g.canonical_dispersion(3.0)
        assert float(cd) == pytest.approx(3.0), f"canonical_dispersion expected 3.0, got {cd}"

    def test_estimate_dispersion_rss_over_df(self):
        """estimate_dispersion(X, y, eta) must return RSS / (n - p)."""
        g = Gaussian()
        n, p = 10, 2
        X = jnp.ones((n, p))
        y = jnp.full(n, 3.0)
        eta = jnp.full(n, 2.0)  # mu = 2.0 via identity link, residual = 1.0 each
        ed = g.estimate_dispersion(X, y, eta)
        expected = jnp.sum((y - 2.0) ** 2) / (n - p)  # 10 / 8 = 1.25
        assert jnp.allclose(ed, expected), f"estimate_dispersion expected {expected}, got {ed}"

    def test_estimate_dispersion_is_finite(self):
        """estimate_dispersion must return a finite scalar."""
        g = Gaussian()
        X = jnp.ones((10, 2))
        y = jnp.ones(10) * 2.0
        eta = jnp.ones(10) * 2.0
        ed = g.estimate_dispersion(X, y, eta)
        assert jnp.isfinite(ed), "estimate_dispersion must be finite"

    def test_estimate_dispersion_zero_residual(self):
        """When mu == y exactly, estimate_dispersion returns 0."""
        g = Gaussian()
        X = jnp.ones((6, 2))
        y = jnp.ones(6)
        eta = jnp.ones(6)  # identity link: mu = eta = 1.0 = y
        ed = g.estimate_dispersion(X, y, eta)
        assert jnp.allclose(ed, 0.0), f"Expected 0 residual dispersion, got {ed}"
