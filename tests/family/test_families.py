# pattern: Functional Core
"""Consolidated exponential family tests.

Groups:
- ExponentialFamily interface (bounds, calc_weight/working_weights signature, sample)
- Gaussian numerics (variance sentinel, dispersion, saturated design)
- Dispersion/auxiliary split semantics
- NegativeBinomial numerics (stability, large eta overflow, gradient)
- Transform safety (JIT/VMAP/AD for all families)
- Sampling (shapes, types, finiteness)
- Canonical dispersion values
- Gamma numerics (bounds, link, variance, dispersion, fit)
"""

import pytest

import jax
import jax.numpy as jnp
import jax.random as jr

import glmax

from glmax import GLMData
from glmax.family import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson


_ALL_FAMILIES = [Gaussian, Poisson, Binomial, NegativeBinomial]
_ALL_FAMILIES_INCLUDING_GAMMA = [*_ALL_FAMILIES, Gamma]
_SPLIT_FAMILY_CASES = [
    (Gaussian, 0.5, 0.2),
    (Poisson, 7.0, 0.2),
    (Binomial, 7.0, 0.2),
    (NegativeBinomial, 1.0, 0.5),
]
_KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# ExponentialFamily interface: bounds
# ---------------------------------------------------------------------------


class TestBounds:
    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES_INCLUDING_GAMMA)
    def test_bounds_is_two_tuple(self, FamilyCls):
        f = FamilyCls()
        assert len(f._bounds) == 2, f"{FamilyCls.__name__}._bounds must be a two-tuple"

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES_INCLUDING_GAMMA)
    def test_bounds_lower_less_than_upper(self, FamilyCls):
        f = FamilyCls()
        lo, hi = f._bounds
        assert lo < hi, f"{FamilyCls.__name__}._bounds lower must be < upper"


# ---------------------------------------------------------------------------
# ExponentialFamily interface: calc_weight / working_weights signature
# ---------------------------------------------------------------------------


class TestCalcWeight:
    def test_gaussian_calc_weight_two_args_returns_three_tuple(self):
        f = Gaussian()
        eta = jnp.zeros(5)
        result = f.calc_weight(eta, 1.0)
        assert len(result) == 3

    def test_gaussian_calc_weight_shapes(self):
        f = Gaussian()
        eta = jnp.zeros(5)
        mu, variance, weight = f.calc_weight(eta, 1.0)
        assert mu.shape == (5,)
        assert variance.shape == (5,)
        assert weight.shape == (5,)

    @pytest.mark.parametrize(("FamilyCls", "disp", "aux"), _SPLIT_FAMILY_CASES)
    def test_all_families_calc_weight_split_disp_aux(self, FamilyCls, disp, aux):
        f = FamilyCls()
        eta = jnp.zeros(5)
        mu, variance, weight = f.calc_weight(eta, disp=disp, aux=aux)
        assert mu.shape == (5,)
        assert variance.shape == (5,)
        assert weight.shape == (5,)

    def test_calc_weight_returns_variance_not_g_deriv(self):
        """Second return value must be variance (V(mu)), not g'(mu).

        For Poisson with log link at eta=1: mu=e, variance=e, g_deriv=1/e.
        They are distinct, so we can confirm v == variance != g_deriv.
        """
        f = Poisson()
        eta = jnp.ones(3)  # mu = exp(1) = e
        mu, v, w = f.calc_weight(eta, 0.0)
        expected_mu = jnp.exp(jnp.ones(3))
        # For Poisson: variance(mu) = mu = e, g_deriv(mu) = 1/mu = 1/e
        assert jnp.allclose(
            v, expected_mu, rtol=1e-5
        ), f"Second return value should be variance=mu=e, not g_deriv=1/e; got {v}"

    def test_calc_weight_four_args_raises_type_error(self):
        f = Gaussian()
        X = jnp.zeros((5, 3))
        y = jnp.zeros(5)
        eta = jnp.zeros(5)
        with pytest.raises(TypeError):
            f.calc_weight(X, y, eta, 1.0)

    @pytest.mark.parametrize(("FamilyCls", "disp", "aux"), _SPLIT_FAMILY_CASES)
    def test_all_families_negloglikelihood_split_disp_aux_signature(self, FamilyCls, disp, aux):
        """negloglikelihood(y, eta, disp, aux) accepted by all families."""
        f = FamilyCls()
        y = jnp.array([1.0, 0.0, 1.0])
        eta = jnp.array([0.5, -0.5, 0.5])
        nll = f.negloglikelihood(y, eta, disp=disp, aux=aux)
        assert jnp.isfinite(nll)


# ---------------------------------------------------------------------------
# Gaussian numerics: variance sentinel, dispersion, saturated design
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
        assert jnp.allclose(v, disp * jnp.ones(4)), f"disp=1e-15 expected {disp}, got {v}"

    def test_variance_negative_disp_returns_ones(self):
        """Gaussian.variance(mu, -1.0) treats negative as sentinel, returns 1.0."""
        g = Gaussian()
        mu = jnp.ones(3)
        v = g.variance(mu, -1.0)
        assert jnp.allclose(v, jnp.ones(3)), f"disp=-1.0 expected sentinel 1.0, got {v}"


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

    def test_update_nuisance_rss_over_df(self):
        """update_nuisance(X, y, eta) must return (RSS / (n - p), None)."""
        g = Gaussian()
        n, p = 10, 2
        X = jnp.ones((n, p))
        y = jnp.full(n, 3.0)
        eta = jnp.full(n, 2.0)  # mu = 2.0 via identity link, residual = 1.0 each
        new_disp, new_aux = g.update_nuisance(X, y, eta, disp=1.0)
        expected = jnp.sum((y - 2.0) ** 2) / (n - p)  # 10 / 8 = 1.25
        assert jnp.allclose(new_disp, expected), f"update_nuisance expected {expected}, got {new_disp}"
        assert new_aux is None

    def test_update_nuisance_is_finite(self):
        """update_nuisance disp must be a finite scalar."""
        g = Gaussian()
        X = jnp.ones((10, 2))
        y = jnp.ones(10) * 2.0
        eta = jnp.ones(10) * 2.0
        new_disp, _ = g.update_nuisance(X, y, eta, disp=1.0)
        assert jnp.isfinite(new_disp), "update_nuisance disp must be finite"

    def test_update_nuisance_zero_residual(self):
        """When mu == y exactly, update_nuisance returns (0, None)."""
        g = Gaussian()
        X = jnp.ones((6, 2))
        y = jnp.ones(6)
        eta = jnp.ones(6)  # identity link: mu = eta = 1.0 = y
        new_disp, new_aux = g.update_nuisance(X, y, eta, disp=1.0)
        assert jnp.allclose(new_disp, 0.0), f"Expected 0 residual dispersion, got {new_disp}"
        assert new_aux is None


class TestGaussianSaturatedDesign:
    """update_nuisance must return finite, non-negative disp even when n == p
    (saturated design, df = n - p = 0)."""

    def test_update_nuisance_saturated_is_finite(self):
        g = Gaussian()
        n = p = 4
        X = jnp.eye(n, p)
        y = jnp.array([1.0, 2.0, 3.0, 4.0])
        eta = y  # identity link: mu == y, RSS == 0 when perfect fit — still must be finite
        result, _ = g.update_nuisance(X, y, eta, disp=1.0)
        assert jnp.isfinite(result), f"update_nuisance saturated: expected finite, got {result}"

    def test_update_nuisance_saturated_is_non_negative(self):
        g = Gaussian()
        n = p = 4
        X = jnp.eye(n, p)
        y = jnp.array([1.0, 2.0, 3.0, 4.0])
        eta = jnp.zeros(n)  # non-zero RSS to make dispersion positive
        result, _ = g.update_nuisance(X, y, eta, disp=1.0)
        assert result >= 0.0, f"update_nuisance saturated: expected >= 0, got {result}"

    def test_update_nuisance_overparameterised_is_finite(self):
        """n < p (more params than observations) must also be finite."""
        g = Gaussian()
        n, p = 3, 5
        X = jnp.ones((n, p))
        y = jnp.ones(n)
        eta = jnp.ones(n)
        result, _ = g.update_nuisance(X, y, eta, disp=1.0)
        assert jnp.isfinite(result), f"update_nuisance n<p: expected finite, got {result}"


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
# NegativeBinomial numerics
# ---------------------------------------------------------------------------


class TestNBNegloglikelihoodStability:
    def test_nb_nll_finite_for_large_y(self):
        nb = NegativeBinomial()
        y = jnp.array([1000.0])
        eta = jnp.log(y)
        nll = nb.negloglikelihood(y, eta, disp=1.0, aux=0.1)
        assert jnp.isfinite(nll), f"NB nll must be finite for large y, got {nll}"

    def test_nb_nll_split_disp_aux_signature(self):
        """negloglikelihood(y, eta, disp, aux) uses aux as alpha."""
        nb = NegativeBinomial()
        y = jnp.array([2.0, 3.0, 1.0])
        eta = jnp.array([0.7, 1.1, 0.3])
        nll = nb.negloglikelihood(y, eta, disp=1.0, aux=0.5)
        assert jnp.isfinite(nll)

    def test_nb_nll_ignores_disp_and_uses_aux(self):
        nb = NegativeBinomial()
        y = jnp.array([2.0, 3.0, 1.0])
        eta = jnp.array([0.7, 1.1, 0.3])

        baseline = nb.negloglikelihood(y, eta, disp=1.0, aux=0.5)
        ignored_disp = nb.negloglikelihood(y, eta, disp=9.0, aux=0.5)
        changed_aux = nb.negloglikelihood(y, eta, disp=1.0, aux=0.2)

        assert jnp.allclose(ignored_disp, baseline)
        assert not jnp.allclose(changed_aux, baseline)

    @pytest.mark.parametrize(
        "method_name",
        ["negloglikelihood", "variance", "sample", "update_nuisance"],
    )
    @pytest.mark.parametrize("bad_aux", [0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])
    def test_nb_invalid_aux_is_rejected_by_direct_numeric_methods(self, method_name, bad_aux):
        nb = NegativeBinomial()
        X = jnp.ones((3, 1))
        y = jnp.array([2.0, 3.0, 1.0])
        eta = jnp.array([0.7, 1.1, 0.3])
        mu = jnp.exp(eta)

        calls = {
            "negloglikelihood": lambda value: nb.negloglikelihood(y, eta, disp=1.0, aux=value),
            "variance": lambda value: nb.variance(mu, disp=1.0, aux=value),
            "sample": lambda value: nb.sample(_KEY, eta, disp=1.0, aux=value),
            "update_nuisance": lambda value: nb.update_nuisance(X, y, eta, disp=1.0, aux=value),
        }

        with pytest.raises(ValueError, match="alpha"):
            calls[method_name](bad_aux)

    @pytest.mark.parametrize(
        "method_name",
        ["negloglikelihood", "variance", "sample", "update_nuisance"],
    )
    @pytest.mark.parametrize("bad_disp", [0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])
    def test_nb_invalid_legacy_disp_is_rejected_by_direct_numeric_methods(self, method_name, bad_disp):
        nb = NegativeBinomial()
        X = jnp.ones((3, 1))
        y = jnp.array([2.0, 3.0, 1.0])
        eta = jnp.array([0.7, 1.1, 0.3])
        mu = jnp.exp(eta)

        calls = {
            "negloglikelihood": lambda value: nb.negloglikelihood(y, eta, disp=value),
            "variance": lambda value: nb.variance(mu, disp=value),
            "sample": lambda value: nb.sample(_KEY, eta, disp=value),
            "update_nuisance": lambda value: nb.update_nuisance(X, y, eta, disp=value),
        }

        with pytest.raises(ValueError, match="alpha"):
            calls[method_name](bad_disp)


class TestNBLargeEtaOverflow:
    def test_nb_nll_finite_for_large_eta(self):
        """NB NLL must be finite for eta=800 (exp(800) overflows float64)."""
        nb = NegativeBinomial()
        y = jnp.array([1000.0])
        eta = jnp.array([800.0])
        nll = nb.negloglikelihood(y, eta, disp=1.0, aux=0.1)
        assert jnp.isfinite(nll), f"NB NLL for eta=800 must be finite, got {nll}"

    def test_nb_nll_finite_for_eta_710(self):
        """NB NLL must be finite at the float64 exp-overflow boundary ~710."""
        nb = NegativeBinomial()
        y = jnp.array([500.0])
        eta = jnp.array([710.0])
        nll = nb.negloglikelihood(y, eta, disp=1.0, aux=0.5)
        assert jnp.isfinite(nll), f"NB NLL for eta=710 must be finite, got {nll}"


class TestNBNegloglikelihoodGradient:
    """Finite-difference verification for NegativeBinomial.negloglikelihood gradients."""

    def test_fd_grad_wrt_aux(self):
        """AD gradient w.r.t. aux must match central finite-difference to rtol=1e-3."""
        nb = NegativeBinomial()
        y = jnp.array([3.0])
        eta = jnp.array([1.0])
        aux = jnp.asarray(0.1)

        grad_aux = jax.grad(lambda a: nb.negloglikelihood(y, eta, disp=1.0, aux=a))(aux)

        eps = 1e-5
        fd_grad = (
            nb.negloglikelihood(y, eta, disp=1.0, aux=aux + eps) - nb.negloglikelihood(y, eta, disp=1.0, aux=aux - eps)
        ) / (2 * eps)

        assert jnp.allclose(
            grad_aux, fd_grad, rtol=1e-3
        ), f"AD grad w.r.t. aux {float(grad_aux):.6g} differs from FD {float(fd_grad):.6g}"

    def test_fd_grad_wrt_eta(self):
        """AD gradient w.r.t. eta (scalar) must match central finite-difference to rtol=1e-3."""
        nb = NegativeBinomial()
        y = jnp.array([3.0])
        aux = 0.1
        eta0 = jnp.array([1.0])

        # grad w.r.t. eta (scalar sum, argnums=1)
        grad_eta = jax.grad(lambda e: nb.negloglikelihood(y, e, disp=1.0, aux=aux))(eta0)

        eps = 1e-5
        fd_grad = (
            nb.negloglikelihood(y, eta0 + eps, disp=1.0, aux=aux)
            - nb.negloglikelihood(y, eta0 - eps, disp=1.0, aux=aux)
        ) / (2 * eps)

        assert jnp.allclose(
            grad_eta, fd_grad, rtol=1e-3
        ), f"AD grad w.r.t. eta {float(grad_eta):.6g} differs from FD {float(fd_grad):.6g}"


# ---------------------------------------------------------------------------
# Transform safety (JIT/VMAP/AD for all families)
# ---------------------------------------------------------------------------


class TestTransformSafety:
    """All 4 families must be JIT-traceable on the key numerics surfaces."""

    @pytest.mark.parametrize(("FamilyCls", "disp", "aux"), _SPLIT_FAMILY_CASES)
    def test_jit_calc_weight(self, FamilyCls, disp, aux):
        """jax.jit(family.calc_weight)(eta, disp, aux) must trace without error."""
        f = FamilyCls()
        eta = jnp.zeros(5)
        mu, v, w = jax.jit(lambda eta_: f.calc_weight(eta_, disp=disp, aux=aux))(eta)
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
        nll = jax.jit(lambda y_, eta_, aux_: nb.negloglikelihood(y_, eta_, disp=1.0, aux=aux_))(y, eta, 0.1)
        assert jnp.isfinite(nll), f"JIT NB NLL must be finite, got {nll}"

    def test_ad_nb_negloglikelihood_wrt_aux(self):
        """jax.grad of NB NLL w.r.t. aux must not raise."""
        nb = NegativeBinomial()
        y = jnp.array([3.0, 5.0, 1.0])
        eta = jnp.array([1.0, 1.5, 0.5])
        aux = jnp.asarray(0.1)
        grad_fn = jax.grad(lambda aux_: nb.negloglikelihood(y, eta, disp=1.0, aux=aux_))
        g = grad_fn(aux)
        assert jnp.isfinite(g), f"AD through NB NLL w.r.t. aux must be finite, got {g}"

    @pytest.mark.parametrize(("FamilyCls", "disp", "aux"), _SPLIT_FAMILY_CASES)
    def test_vmap_negloglikelihood(self, FamilyCls, disp, aux):
        """jax.vmap over a batch of scalar etas must produce finite outputs for all families."""
        f = FamilyCls()
        eta_scalar_batch = jnp.linspace(-2.0, 2.0, 8)
        result = jax.vmap(lambda e: f.negloglikelihood(jnp.array([1.0]), jnp.array([e]), disp=disp, aux=aux))(
            eta_scalar_batch
        )
        assert result.shape == (8,), f"{FamilyCls.__name__}: expected shape (8,), got {result.shape}"
        assert jnp.all(
            jnp.isfinite(result)
        ), f"{FamilyCls.__name__}: vmap negloglikelihood produced non-finite values: {result}"

    @pytest.mark.parametrize(("FamilyCls", "disp", "aux"), _SPLIT_FAMILY_CASES)
    def test_vmap_sample(self, FamilyCls, disp, aux):
        """jax.vmap over a batch of scalar etas must produce finite samples for all families."""
        f = FamilyCls()
        eta_scalar_batch = jnp.linspace(-2.0, 2.0, 8)
        keys = jax.random.split(_KEY, 8)
        result = jax.vmap(lambda key, e: f.sample(key, jnp.array([e]), disp=disp, aux=aux))(keys, eta_scalar_batch)
        assert result.shape == (8, 1), f"{FamilyCls.__name__}: expected shape (8, 1), got {result.shape}"
        assert jnp.all(jnp.isfinite(result)), f"{FamilyCls.__name__}: vmap sample produced non-finite values: {result}"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSample:
    @pytest.mark.parametrize(("FamilyCls", "disp", "aux"), _SPLIT_FAMILY_CASES)
    def test_sample_returns_correct_shape(self, FamilyCls, disp, aux):
        """sample(key, eta, disp, aux) returns array with shape == eta.shape."""
        f = FamilyCls()
        eta = jnp.zeros(10)
        s = f.sample(_KEY, eta, disp=disp, aux=aux)
        assert s.shape == (10,), f"{FamilyCls.__name__}.sample shape {s.shape} != (10,)"

    @pytest.mark.parametrize(("FamilyCls", "disp", "aux"), _SPLIT_FAMILY_CASES)
    def test_sample_returns_jax_array(self, FamilyCls, disp, aux):
        """sample must return a JAX array (not numpy)."""
        f = FamilyCls()
        eta = jnp.zeros(5)
        s = f.sample(_KEY, eta, disp=disp, aux=aux)
        assert isinstance(s, jax.Array), f"{FamilyCls.__name__}.sample must return a JAX array"

    @pytest.mark.parametrize(("FamilyCls", "disp", "aux"), _SPLIT_FAMILY_CASES)
    def test_sample_is_finite(self, FamilyCls, disp, aux):
        """Samples must be finite (no NaN/Inf)."""
        f = FamilyCls()
        eta = jnp.zeros(20)
        s = f.sample(_KEY, eta, disp=disp, aux=aux)
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
        s = nb.sample(_KEY, eta, disp=1.0, aux=0.5)
        assert s.shape == (100,)
        assert jnp.all(s >= 0), "NB samples must be non-negative"

    def test_nb_sample_ignores_disp_and_uses_aux(self):
        nb = NegativeBinomial()
        eta = jnp.full(100, jnp.log(5.0))

        baseline = nb.sample(_KEY, eta, disp=1.0, aux=0.5)
        ignored_disp = nb.sample(_KEY, eta, disp=9.0, aux=0.5)
        changed_aux = nb.sample(_KEY, eta, disp=1.0, aux=0.2)

        assert jnp.array_equal(ignored_disp, baseline)
        assert not jnp.array_equal(changed_aux, baseline)

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
# Canonical dispersion values
# ---------------------------------------------------------------------------


def test_init_nuisance_returns_correct_defaults() -> None:
    for family in [Gaussian(), Poisson(), Binomial(), Gamma()]:
        disp, aux = family.init_nuisance()
        assert jnp.allclose(disp, jnp.asarray(1.0))
        assert aux is None

    nb_disp, nb_aux = NegativeBinomial().init_nuisance()
    assert jnp.allclose(nb_disp, jnp.asarray(1.0))
    assert nb_aux is not None
    assert float(nb_aux) > 0.0


class TestFamilyDocstrings:
    def test_docstrings_describe_disp_aux_split(self):
        assert "uses `disp` as EDM dispersion and ignores `aux`" in Gaussian.__doc__
        assert "fixes `disp = 1.0` and ignores `aux`" in Poisson.__doc__
        assert "fixes `disp = 1.0` and ignores `aux`" in Binomial.__doc__
        assert "fixes `disp = 1.0` and uses `aux` as `alpha`" in NegativeBinomial.__doc__
        assert "uses `disp` as EDM dispersion and ignores `aux`" in Gamma.__doc__


# ---------------------------------------------------------------------------
# Gamma numerics
# ---------------------------------------------------------------------------


class TestGamma:
    def test_bounds_lower_is_positive(self):
        """Gamma lower bound must be strictly positive (not zero)."""
        g = Gamma()
        assert g._bounds[0] > 0

    def test_default_link_is_inverse(self):
        from glmax.family import InverseLink

        assert isinstance(Gamma().glink, InverseLink)

    def test_variance(self):
        """Gamma.variance(mu, disp, aux) == disp * mu**2 and ignores aux."""
        g = Gamma()
        mu = jnp.array([1.0, 2.0, 3.0])
        v = g.variance(mu, disp=0.5, aux=0.3)
        assert jnp.allclose(v, 0.5 * mu**2)

    def test_negloglikelihood_finite(self):
        g = Gamma()
        y = jnp.ones(20) * 2.0
        eta = jnp.ones(20) * 0.5  # InverseLink: mu = 1/eta = 2
        assert jnp.isfinite(g.negloglikelihood(y, eta, disp=1.0, aux=0.3))

    def test_sample_shape_and_positivity(self):
        g = Gamma()
        key = jr.PRNGKey(0)
        eta = jnp.ones(20)  # InverseLink: mu = 1.0
        s = g.sample(key, eta, disp=1.0, aux=0.3)
        assert s.shape == (20,)
        assert jnp.all(s > 0), "Gamma samples must be positive"

    def test_update_nuisance_passthrough(self):
        g = Gamma()
        result_disp, result_aux = g.update_nuisance(jnp.zeros((5, 3)), jnp.ones(5), jnp.ones(5) * 0.5, disp=1.0)
        assert float(result_disp) == 1.0
        assert result_aux is None

    def test_fits_positive_response_dataset(self):
        key = jr.PRNGKey(42)
        key_X, key_y = jr.split(key)
        n, p = 50, 3
        X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key_X, (n, p - 1))], axis=1)
        y = jr.gamma(key_y, 2.0, shape=(n,))
        result = glmax.fit(glmax.specify(family=Gamma()), GLMData(X=X, y=y))
        assert jnp.all(jnp.isfinite(result.params.beta))
        assert jnp.isfinite(result.params.disp)
