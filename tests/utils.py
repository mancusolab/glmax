import numpy.testing as nptest

from jax.typing import ArrayLike


def assert_array_eq(estimate: ArrayLike, truth: ArrayLike, rtol=1e-7, atol=1e-8):
    nptest.assert_allclose(estimate, truth, rtol=rtol, atol=atol)


def assert_glm_state_parity(wrapper_state, direct_state, rtol=1e-7, atol=1e-8):
    assert_array_eq(wrapper_state.beta, direct_state.beta, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.se, direct_state.se, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.z, direct_state.z, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.p, direct_state.p, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.eta, direct_state.eta, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.mu, direct_state.mu, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.glm_wt, direct_state.glm_wt, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.infor_inv, direct_state.infor_inv, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.resid, direct_state.resid, rtol=rtol, atol=atol)
    assert_array_eq(wrapper_state.alpha, direct_state.alpha, rtol=rtol, atol=atol)

    assert int(wrapper_state.num_iters) == int(direct_state.num_iters)
    assert bool(wrapper_state.converged) is bool(direct_state.converged)
