# pattern: Functional Core
"""Tests for Subcomponent A: _bounds, calc_weight signature, negloglikelihood signature.

Covers:
- expfam-port.AC2.1: all four families have _bounds two-tuples
- expfam-port.AC2.2: calc_weight(eta, disp) returns (mu, variance, weight) three-tuple
- expfam-port.AC2.4: NB negloglikelihood is finite for large y
- expfam-port.AC2.7: calc_weight(X, y, eta, disp) with four positional args raises TypeError
"""
import pytest

import jax.numpy as jnp

from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


_ALL_FAMILIES = [Gaussian, Poisson, Binomial, NegativeBinomial]


# --- AC2.1: _bounds two-tuples ---


class TestBounds:
    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_bounds_is_two_tuple(self, FamilyCls):
        f = FamilyCls()
        assert len(f._bounds) == 2, f"{FamilyCls.__name__}._bounds must be a two-tuple"

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_bounds_lower_less_than_upper(self, FamilyCls):
        f = FamilyCls()
        lo, hi = f._bounds
        assert lo < hi, f"{FamilyCls.__name__}._bounds lower must be < upper"


# --- AC2.2: calc_weight(eta, disp) returns (mu, variance, weight) ---


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

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_all_families_calc_weight_two_args(self, FamilyCls):
        f = FamilyCls()
        eta = jnp.zeros(5)
        mu, variance, weight = f.calc_weight(eta, 0.1)
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


# --- AC2.7: four positional args raises TypeError ---


class TestCalcWeightFourArgRaises:
    def test_gaussian_four_args_raises_type_error(self):
        f = Gaussian()
        X = jnp.zeros((5, 3))
        y = jnp.zeros(5)
        eta = jnp.zeros(5)
        with pytest.raises(TypeError):
            f.calc_weight(X, y, eta, 1.0)


# --- AC2.4: NB negloglikelihood finite for large y ---


class TestNBNegloglikelihoodStability:
    def test_nb_nll_finite_for_large_y(self):
        nb = NegativeBinomial()
        y = jnp.array([1000.0])
        eta = jnp.log(y)
        nll = nb.negloglikelihood(y, eta, 0.1)
        assert jnp.isfinite(nll), f"NB nll must be finite for large y, got {nll}"

    def test_nb_nll_new_signature_two_arg(self):
        """negloglikelihood(y, eta, disp) — no X."""
        nb = NegativeBinomial()
        y = jnp.array([2.0, 3.0, 1.0])
        eta = jnp.array([0.7, 1.1, 0.3])
        nll = nb.negloglikelihood(y, eta, 0.5)
        assert jnp.isfinite(nll)

    @pytest.mark.parametrize("FamilyCls", _ALL_FAMILIES)
    def test_all_families_negloglikelihood_new_signature(self, FamilyCls):
        """negloglikelihood(y, eta, disp) accepted by all families."""
        f = FamilyCls()
        y = jnp.array([1.0, 0.0, 1.0])
        eta = jnp.array([0.5, -0.5, 0.5])
        nll = f.negloglikelihood(y, eta, 0.5)
        assert jnp.isfinite(nll)
