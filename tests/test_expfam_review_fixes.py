# pattern: Functional Core
"""Regression tests for Phase 3 numerics-interface-auditor + scientific-inference-algorithm-reviewer fixes.

Covers:
- High 1: Gaussian.update_dispersion / estimate_dispersion df<=0 guard (saturated design)
- High 2: JIT/VMAP/AD transform-safety for all 4 family surfaces
- Medium 1: NB negloglikelihood finite for large eta (overflow guard)
- Medium 3: Gaussian.variance sentinel — only literal 0.0 triggers fallback
- Low 1: Gaussian.sample disp<=0 guard
- Low 2: Poisson.canonical_dispersion == 1.0 and Binomial.canonical_dispersion == 1.0
"""
import pytest

import jax
import jax.numpy as jnp

from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


_ALL_FAMILIES = [Gaussian, Poisson, Binomial, NegativeBinomial]
_KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# High 1: Gaussian df<=0 guard (saturated design)
# ---------------------------------------------------------------------------


class TestGaussianSaturatedDesign:
    """update_dispersion and estimate_dispersion must return finite, non-negative
    results even when n == p (saturated design, df = n - p = 0)."""

    def test_update_dispersion_saturated_is_finite(self):
        g = Gaussian()
        n = p = 4
        X = jnp.eye(n, p)
        y = jnp.array([1.0, 2.0, 3.0, 4.0])
        eta = y  # identity link: mu == y, RSS == 0 when perfect fit — still must be finite
        result = g.update_dispersion(X, y, eta)
        assert jnp.isfinite(result), f"update_dispersion saturated: expected finite, got {result}"

    def test_update_dispersion_saturated_is_non_negative(self):
        g = Gaussian()
        n = p = 4
        X = jnp.eye(n, p)
        y = jnp.array([1.0, 2.0, 3.0, 4.0])
        eta = jnp.zeros(n)  # non-zero RSS to make dispersion positive
        result = g.update_dispersion(X, y, eta)
        assert result >= 0.0, f"update_dispersion saturated: expected >= 0, got {result}"

    def test_estimate_dispersion_saturated_is_finite(self):
        g = Gaussian()
        n = p = 5
        X = jnp.eye(n, p)
        y = jnp.ones(n) * 3.0
        eta = jnp.zeros(n)  # non-zero residuals
        result = g.estimate_dispersion(X, y, eta)
        assert jnp.isfinite(result), f"estimate_dispersion saturated: expected finite, got {result}"

    def test_estimate_dispersion_saturated_is_non_negative(self):
        g = Gaussian()
        n = p = 5
        X = jnp.eye(n, p)
        y = jnp.ones(n) * 3.0
        eta = jnp.zeros(n)
        result = g.estimate_dispersion(X, y, eta)
        assert result >= 0.0, f"estimate_dispersion saturated: expected >= 0, got {result}"

    def test_update_dispersion_overparameterised_is_finite(self):
        """n < p (more params than observations) must also be finite."""
        g = Gaussian()
        n, p = 3, 5
        X = jnp.ones((n, p))
        y = jnp.ones(n)
        eta = jnp.ones(n)
        result = g.update_dispersion(X, y, eta)
        assert jnp.isfinite(result), f"update_dispersion n<p: expected finite, got {result}"


# ---------------------------------------------------------------------------
# High 2: JIT/VMAP/AD transform-safety
# ---------------------------------------------------------------------------


class TestTransformSafety:
    """All 4 families must be JIT-traceable on the key numerics surfaces."""

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_jit_calc_weight(self, FamilyCls):
        """jax.jit(family.calc_weight)(eta, disp) must trace without error."""
        f = FamilyCls()
        eta = jnp.zeros(5)
        disp = 0.1
        mu, v, w = jax.jit(f.calc_weight)(eta, disp)
        assert mu.shape == (5,)
        assert v.shape == (5,)
        assert w.shape == (5,)

    def test_jit_gaussian_variance_sentinel_zero(self):
        """Gaussian.variance(mu, 0.0) sentinel path must trace under JIT."""
        g = Gaussian()
        mu = jnp.ones(4)
        v = jax.jit(g.variance)(mu, 0.0)
        assert v.shape == (4,)
        assert jnp.allclose(v, jnp.ones(4)), f"sentinel 0.0 → fallback 1.0, got {v}"

    def test_jit_gaussian_variance_nonzero_disp(self):
        """Gaussian.variance(mu, 2.0) must trace under JIT and return 2.0."""
        g = Gaussian()
        mu = jnp.ones(4)
        v = jax.jit(g.variance)(mu, 2.0)
        assert jnp.allclose(v, 2.0 * jnp.ones(4)), f"disp=2.0 expected 2.0, got {v}"

    def test_jit_nb_negloglikelihood_logaddexp_path(self):
        """NegativeBinomial.negloglikelihood must JIT-trace (logaddexp path)."""
        nb = NegativeBinomial()
        y = jnp.array([3.0, 5.0, 1.0])
        eta = jnp.array([1.0, 1.5, 0.5])
        nll = jax.jit(nb.negloglikelihood)(y, eta, 0.1)
        assert jnp.isfinite(nll), f"JIT NB NLL must be finite, got {nll}"

    def test_ad_nb_negloglikelihood_wrt_disp(self):
        """jax.grad of NB NLL w.r.t. disp (argnums=2) must not raise."""
        nb = NegativeBinomial()
        y = jnp.array([3.0, 5.0, 1.0])
        eta = jnp.array([1.0, 1.5, 0.5])
        disp = jnp.asarray(0.1)
        grad_fn = jax.grad(nb.negloglikelihood, argnums=2)
        g = grad_fn(y, eta, disp)
        assert jnp.isfinite(g), f"AD through NB NLL w.r.t. disp must be finite, got {g}"

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_vmap_negloglikelihood(self, FamilyCls):
        """jax.vmap over a batch of scalar etas must produce finite outputs for all families."""
        f = FamilyCls()
        eta_scalar_batch = jnp.linspace(-2.0, 2.0, 8)
        disp = 0.1
        result = jax.vmap(lambda e: f.negloglikelihood(jnp.array([1.0]), jnp.array([e]), disp))(eta_scalar_batch)
        assert result.shape == (8,), f"{FamilyCls.__name__}: expected shape (8,), got {result.shape}"
        assert jnp.all(
            jnp.isfinite(result)
        ), f"{FamilyCls.__name__}: vmap negloglikelihood produced non-finite values: {result}"

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_vmap_sample(self, FamilyCls):
        """jax.vmap over a batch of scalar etas must produce finite samples for all families."""
        f = FamilyCls()
        eta_scalar_batch = jnp.linspace(-2.0, 2.0, 8)
        disp = 0.1
        keys = jax.random.split(_KEY, 8)
        result = jax.vmap(lambda key, e: f.sample(key, jnp.array([e]), disp))(keys, eta_scalar_batch)
        assert result.shape == (8, 1), f"{FamilyCls.__name__}: expected shape (8, 1), got {result.shape}"
        assert jnp.all(jnp.isfinite(result)), f"{FamilyCls.__name__}: vmap sample produced non-finite values: {result}"


# ---------------------------------------------------------------------------
# Low: NB negloglikelihood finite-difference gradient check
# ---------------------------------------------------------------------------


class TestNBNegloglikelihoodGradient:
    """Finite-difference verification for NegativeBinomial.negloglikelihood gradients."""

    def test_fd_grad_wrt_disp(self):
        """AD gradient w.r.t. disp must match central finite-difference to rtol=1e-3."""
        nb = NegativeBinomial()
        y = jnp.array([3.0])
        eta = jnp.array([1.0])
        disp = jnp.asarray(0.1)

        grad_disp = jax.grad(lambda d: nb.negloglikelihood(y, eta, d))(disp)

        eps = 1e-5
        fd_grad = (nb.negloglikelihood(y, eta, disp + eps) - nb.negloglikelihood(y, eta, disp - eps)) / (2 * eps)

        assert jnp.allclose(
            grad_disp, fd_grad, rtol=1e-3
        ), f"AD grad w.r.t. disp {float(grad_disp):.6g} differs from FD {float(fd_grad):.6g}"

    def test_fd_grad_wrt_eta(self):
        """AD gradient w.r.t. eta (scalar) must match central finite-difference to rtol=1e-3."""
        nb = NegativeBinomial()
        y = jnp.array([3.0])
        disp = 0.1
        eta0 = jnp.array([1.0])

        # grad w.r.t. eta (scalar sum, argnums=1)
        grad_eta = jax.grad(lambda e: nb.negloglikelihood(y, e, disp))(eta0)

        eps = 1e-5
        fd_grad = (nb.negloglikelihood(y, eta0 + eps, disp) - nb.negloglikelihood(y, eta0 - eps, disp)) / (2 * eps)

        assert jnp.allclose(
            grad_eta, fd_grad, rtol=1e-3
        ), f"AD grad w.r.t. eta {float(grad_eta):.6g} differs from FD {float(fd_grad):.6g}"


# ---------------------------------------------------------------------------
# Medium 1: NB large-eta overflow
# ---------------------------------------------------------------------------


class TestNBLargeEtaOverflow:
    def test_nb_nll_finite_for_large_eta(self):
        """NB NLL must be finite for eta=800 (exp(800) overflows float64)."""
        nb = NegativeBinomial()
        y = jnp.array([1000.0])
        eta = jnp.array([800.0])
        nll = nb.negloglikelihood(y, eta, 0.1)
        assert jnp.isfinite(nll), f"NB NLL for eta=800 must be finite, got {nll}"

    def test_nb_nll_finite_for_eta_710(self):
        """NB NLL must be finite at the float64 exp-overflow boundary ~710."""
        nb = NegativeBinomial()
        y = jnp.array([500.0])
        eta = jnp.array([710.0])
        nll = nb.negloglikelihood(y, eta, 0.5)
        assert jnp.isfinite(nll), f"NB NLL for eta=710 must be finite, got {nll}"


# ---------------------------------------------------------------------------
# Medium 3: Gaussian.variance sentinel behaviour
# ---------------------------------------------------------------------------


class TestGaussianVarianceSentinel:
    def test_variance_zero_disp_returns_ones(self):
        """Gaussian.variance(mu, 0.0) must fall back to 1.0 (sentinel path)."""
        g = Gaussian()
        mu = jnp.zeros(3)
        v = g.variance(mu, 0.0)
        assert jnp.allclose(v, jnp.ones(3)), f"disp=0.0 sentinel expected 1.0, got {v}"

    def test_variance_small_positive_disp_does_not_trigger_fallback(self):
        """Gaussian.variance(mu, 1e-15) must return 1e-15 * ones_like(mu), not 1.0."""
        g = Gaussian()
        mu = jnp.ones(4)
        disp = 1e-15
        v = g.variance(mu, disp)
        # The sentinel is only for disp <= 0; 1e-15 > 0 so no fallback
        assert jnp.allclose(v, disp * jnp.ones(4)), f"disp=1e-15 expected {disp}, got {v}"

    def test_variance_negative_disp_returns_ones(self):
        """Gaussian.variance(mu, -1.0) treats negative as sentinel, returns 1.0."""
        g = Gaussian()
        mu = jnp.ones(3)
        v = g.variance(mu, -1.0)
        assert jnp.allclose(v, jnp.ones(3)), f"disp=-1.0 expected sentinel 1.0, got {v}"


# ---------------------------------------------------------------------------
# Low 1: Gaussian.sample disp <= 0 guard
# ---------------------------------------------------------------------------


class TestGaussianSampleDispGuard:
    def test_sample_zero_disp_returns_finite(self):
        """Gaussian.sample(key, eta, 0.0) must return finite samples."""
        g = Gaussian()
        eta = jnp.zeros(10)
        s = g.sample(_KEY, eta, 0.0)
        assert jnp.all(jnp.isfinite(s)), f"sample(disp=0.0) produced non-finite values: {s}"

    def test_sample_zero_disp_shape_preserved(self):
        """Shape must be preserved even with sentinel disp=0.0."""
        g = Gaussian()
        eta = jnp.zeros(7)
        s = g.sample(_KEY, eta, 0.0)
        assert s.shape == (7,), f"Expected shape (7,), got {s.shape}"

    def test_sample_negative_disp_returns_finite(self):
        """Gaussian.sample(key, eta, -1.0) must return finite samples."""
        g = Gaussian()
        eta = jnp.ones(5)
        s = g.sample(_KEY, eta, -1.0)
        assert jnp.all(jnp.isfinite(s)), f"sample(disp=-1.0) produced non-finite: {s}"


# ---------------------------------------------------------------------------
# Low 2: Poisson and Binomial canonical_dispersion == 1.0
# ---------------------------------------------------------------------------


class TestCanonicalDispersionUnitFamilies:
    def test_poisson_canonical_dispersion_is_one(self):
        """Poisson.canonical_dispersion(any) must return 1.0."""
        p = Poisson()
        assert float(p.canonical_dispersion(0.0)) == pytest.approx(1.0)

    def test_poisson_canonical_dispersion_ignores_argument(self):
        """Poisson.canonical_dispersion(5.0) must still return 1.0."""
        p = Poisson()
        assert float(p.canonical_dispersion(5.0)) == pytest.approx(1.0)

    def test_binomial_canonical_dispersion_is_one(self):
        """Binomial.canonical_dispersion(any) must return 1.0."""
        b = Binomial()
        assert float(b.canonical_dispersion(0.0)) == pytest.approx(1.0)

    def test_binomial_canonical_dispersion_ignores_argument(self):
        """Binomial.canonical_dispersion(3.0) must still return 1.0."""
        b = Binomial()
        assert float(b.canonical_dispersion(3.0)) == pytest.approx(1.0)

    def test_poisson_canonical_dispersion_returns_jax_array(self):
        p = Poisson()
        result = p.canonical_dispersion(0.0)
        assert isinstance(result, jax.Array)

    def test_binomial_canonical_dispersion_returns_jax_array(self):
        b = Binomial()
        result = b.canonical_dispersion(0.0)
        assert isinstance(result, jax.Array)
