# pattern: Functional Core
"""Tests for glmax.family.links — TDD regression tests for numerics audit findings.

Covers:
- [High 1] Primal correctness, JIT round-trips, vmap batch consistency, finite-difference
  gradient checks, and nested-vmap safety for InverseLink and PowerLink.
- [High 2] Nested-vmap safety of _grad_per_sample (eqx.filter_vmap/filter_grad).
- [High 3] InverseLink.inverse(eta=0) returns finite value.
- [Medium 1] NBLink.__call__ precision at large mu.
- [Low 1] PowerLink(0.0) raises ValueError.
"""

import numpy as np
import pytest

import equinox as eqx
import jax
import jax.numpy as jnp

from glmax.family.links import (
    InverseLink,
    NBLink,
    PowerLink,
)


# ---------------------------------------------------------------------------
# [High 1a] Primal correctness for InverseLink
# ---------------------------------------------------------------------------


def test_inverse_link_call_known_value():
    """InverseLink()(2.0) == 0.5."""
    link = InverseLink()
    result = link(jnp.array(2.0))
    np.testing.assert_allclose(result, 0.5, rtol=1e-6)


def test_inverse_link_inverse_known_value():
    """InverseLink().inverse(0.5) == 2.0."""
    link = InverseLink()
    result = link.inverse(jnp.array(0.5))
    np.testing.assert_allclose(result, 2.0, rtol=1e-6)


def test_inverse_link_roundtrip():
    """InverseLink: inverse(call(mu)) == mu for a positive array."""
    link = InverseLink()
    mu = jnp.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(link.inverse(link(mu)), mu, rtol=1e-6)


def test_inverse_link_deriv_known_value():
    """InverseLink.deriv(mu) == -1/mu^2 at mu=2: expect -0.25."""
    link = InverseLink()
    result = link.deriv(jnp.array([2.0]))
    np.testing.assert_allclose(result, jnp.array([-0.25]), rtol=1e-5)


def test_inverse_link_inverse_deriv_known_value():
    """InverseLink.inverse_deriv(eta) == -1/eta^2 at eta=2: expect -0.25."""
    link = InverseLink()
    result = link.inverse_deriv(jnp.array([2.0]))
    np.testing.assert_allclose(result, jnp.array([-0.25]), rtol=1e-5)


# ---------------------------------------------------------------------------
# [High 1b] eqx.filter_jit round-trips for InverseLink and PowerLink
# ---------------------------------------------------------------------------


def test_inverse_link_filter_jit_roundtrip():
    """InverseLink round-trip survives eqx.filter_jit."""
    link = InverseLink()

    @eqx.filter_jit
    def jit_call(lnk, x):
        return lnk(x)

    @eqx.filter_jit
    def jit_inverse(lnk, x):
        return lnk.inverse(x)

    mu = jnp.array([1.0, 2.0, 4.0])
    eta = jit_call(link, mu)
    mu_rt = jit_inverse(link, eta)
    np.testing.assert_allclose(mu_rt, mu, rtol=1e-6)


def test_power_link_filter_jit_roundtrip():
    """PowerLink(2.0) round-trip survives eqx.filter_jit."""
    link = PowerLink(2.0)

    @eqx.filter_jit
    def jit_call(lnk, x):
        return lnk(x)

    @eqx.filter_jit
    def jit_inverse(lnk, x):
        return lnk.inverse(x)

    mu = jnp.array([1.0, 2.0, 3.0])
    eta = jit_call(link, mu)
    mu_rt = jit_inverse(link, eta)
    np.testing.assert_allclose(mu_rt, mu, rtol=1e-5)


# ---------------------------------------------------------------------------
# [High 1c] eqx.filter_vmap batch consistency
# ---------------------------------------------------------------------------


def test_inverse_link_vmap_batch_consistency():
    """InverseLink: batched call matches scalar vmapped result."""
    link = InverseLink()
    mu_batch = jnp.array([1.0, 2.0, 4.0])

    batched = link(mu_batch)
    vmapped = eqx.filter_vmap(lambda x: link(x))(mu_batch)
    np.testing.assert_allclose(batched, vmapped, rtol=1e-7)


def test_power_link_vmap_batch_consistency():
    """PowerLink(2.0): batched call matches scalar vmapped result."""
    link = PowerLink(2.0)
    mu_batch = jnp.array([1.0, 2.0, 3.0])

    batched = link(mu_batch)
    vmapped = eqx.filter_vmap(lambda x: link(x))(mu_batch)
    np.testing.assert_allclose(batched, vmapped, rtol=1e-7)


# ---------------------------------------------------------------------------
# [High 1d] Finite-difference comparison for deriv and inverse_deriv
# ---------------------------------------------------------------------------


def _finite_diff(func, x, eps=1e-5):
    """Central finite difference gradient."""
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def test_inverse_link_deriv_vs_finite_diff():
    """InverseLink.deriv agrees with central finite differences."""
    link = InverseLink()
    mu = jnp.array([0.5, 1.0, 2.0, 5.0])
    ad_deriv = link.deriv(mu)
    fd_deriv = jax.vmap(lambda x: _finite_diff(link, x))(mu)
    np.testing.assert_allclose(ad_deriv, fd_deriv, rtol=1e-4)


def test_power_link_deriv_vs_finite_diff():
    """PowerLink(3.0).deriv agrees with central finite differences."""
    link = PowerLink(3.0)
    mu = jnp.array([0.5, 1.0, 2.0])
    ad_deriv = link.deriv(mu)
    fd_deriv = jax.vmap(lambda x: _finite_diff(link, x))(mu)
    np.testing.assert_allclose(ad_deriv, fd_deriv, rtol=1e-4)


def test_inverse_link_inverse_deriv_vs_finite_diff():
    """InverseLink.inverse_deriv agrees with central finite differences."""
    link = InverseLink()
    eta = jnp.array([0.5, 1.0, 2.0, 5.0])
    ad_deriv = link.inverse_deriv(eta)
    fd_deriv = jax.vmap(lambda x: _finite_diff(link.inverse, x))(eta)
    np.testing.assert_allclose(ad_deriv, fd_deriv, rtol=1e-4)


# ---------------------------------------------------------------------------
# [High 1e / High 2] Nested-vmap test for PowerLink.deriv
# ---------------------------------------------------------------------------


def test_power_link_deriv_nested_vmap():
    """PowerLink.deriv is safe under nested eqx.filter_vmap (High 2 regression)."""
    link = PowerLink(2.0)
    # Shape (2, 3): outer vmap over rows, inner _grad_per_sample vmaps over cols
    mu_2d = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Outer filter_vmap over the batch dimension; inner vmap is inside .deriv
    result = eqx.filter_vmap(lambda row: link.deriv(row))(mu_2d)

    # Expected: d/dmu (mu^2) = 2*mu
    expected = 2.0 * mu_2d
    np.testing.assert_allclose(result, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# [High 3] InverseLink.inverse(eta=0) returns finite value
# ---------------------------------------------------------------------------


def test_inverse_link_inverse_at_zero_is_finite():
    """InverseLink.inverse(0.0) must return a finite value, not inf."""
    link = InverseLink()
    result = link.inverse(jnp.array([0.0]))
    assert jnp.all(jnp.isfinite(result)), f"Expected finite value, got {result}"


# ---------------------------------------------------------------------------
# [Medium 1] NBLink precision at large mu
# ---------------------------------------------------------------------------


def test_nb_link_call_finite_at_large_mu():
    """NBLink(1.0)(mu) is finite across a wide dynamic range."""
    link = NBLink(1.0)
    mu = jnp.array([1e-10, 1.0, 1e10, 1e15])
    result = link(mu)
    assert jnp.all(jnp.isfinite(result)), f"Non-finite values: {result}"


def test_nb_link_call_matches_stable_formula():
    """NBLink(1.0)(mu) matches log(mu) - log1p(mu) for alpha=1 at moderate values."""
    link = NBLink(1.0)
    mu = jnp.array([0.1, 1.0, 10.0, 100.0])
    mu_alpha = mu * 1.0  # alpha=1
    stable_ref = jnp.log(mu_alpha) - jnp.log1p(mu_alpha)
    result = link(mu)
    np.testing.assert_allclose(result, stable_ref, rtol=1e-6)


# ---------------------------------------------------------------------------
# [Low 1] PowerLink(0.0) raises ValueError
# ---------------------------------------------------------------------------


def test_power_link_zero_power_raises():
    """PowerLink(0.0) must raise ValueError: inverse is undefined."""
    with pytest.raises(ValueError, match="power=0"):
        PowerLink(0.0)


# ---------------------------------------------------------------------------
# [Low 2] NBLink.inverse domain boundary: eta=0 returns -inf (documented)
# ---------------------------------------------------------------------------


def test_nb_link_inverse_at_zero_is_neg_inf():
    """NBLink.inverse(0.0) returns -inf.

    The docstring for NBLink.inverse states the domain is entries < 0.
    At eta=0, expm1(-0) = 0, making 1/(alpha*0) = -inf. This test documents
    that boundary behavior explicitly so any future silent-clip guard will
    surface as a deliberate, reviewed change.
    """
    link = NBLink(1.0)
    result = link.inverse(jnp.array([0.0]))
    # Domain constraint: eta=0 is outside the valid domain (eta < 0).
    # The expected mathematical result is -inf; confirm it rather than masking it.
    assert jnp.all(result == -jnp.inf), f"Expected -inf at eta=0, got {result}"
