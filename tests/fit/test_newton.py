# pattern: Imperative Shell

import numpy as np
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose
from statsmodels.genmod.families import links as sm_links

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

import glmax

from glmax import FitResult, FittedGLM
from glmax._fit.newton import NewtonFitter
from glmax.family import Binomial, Gaussian, Poisson
from glmax.family.links import IdentityLink, LogitLink, LogLink


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_KEY = jr.key(0)

_N = 40


def _gaussian_data():
    key1, key2 = jr.split(_KEY)
    X = jnp.column_stack([jnp.ones(_N), jr.normal(key1, (_N,))])
    beta_true = jnp.array([1.5, -0.8])
    y = X @ beta_true + 0.3 * jr.normal(key2, (_N,))
    return Gaussian(IdentityLink()), X, y


def _poisson_data():
    key1, key2 = jr.split(_KEY, 3)[:2]
    X = jnp.column_stack([jnp.ones(_N), jr.normal(key1, (_N,))])
    lam = jnp.exp(X @ jnp.array([0.5, 0.4]))
    y = jr.poisson(key2, lam).astype(jnp.float64)
    return Poisson(LogLink()), X, y


def _binomial_data():
    key1, key2 = jr.split(_KEY, 3)[:2]
    X = jnp.column_stack([jnp.ones(_N), jr.normal(key1, (_N,))])
    p = jax_sigmoid(X @ jnp.array([0.2, 0.6]))
    y = jr.bernoulli(key2, p).astype(jnp.float64)
    return Binomial(LogitLink()), X, y


def jax_sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


# ---------------------------------------------------------------------------
# Construction and interface
# ---------------------------------------------------------------------------


def test_newton_fitter_default_construction():
    fitter = NewtonFitter()
    assert isinstance(fitter.solver, lx.Cholesky)
    assert fitter.step_size == 1.0
    assert fitter.tol == 1e-6
    assert fitter.max_iter == 200
    assert fitter.armijo_c == 0.1
    assert fitter.armijo_factor == 0.5


def test_newton_fitter_custom_construction():
    fitter = NewtonFitter(solver=lx.QR(), tol=1e-8, max_iter=100, armijo_c=0.3, armijo_factor=0.8)
    assert isinstance(fitter.solver, lx.QR)
    assert fitter.tol == 1e-8
    assert fitter.max_iter == 100
    assert fitter.armijo_c == 0.3
    assert fitter.armijo_factor == 0.8


def test_newton_fitter_is_abstract_fitter_subclass():
    assert isinstance(NewtonFitter(), glmax.AbstractFitter)


def test_newton_fitter_exported_from_package_root():
    assert hasattr(glmax, "NewtonFitter")
    assert glmax.NewtonFitter is NewtonFitter


# ---------------------------------------------------------------------------
# Convergence and output shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family_data", [_gaussian_data, _poisson_data, _binomial_data])
def test_newton_fitter_returns_fit_result(family_data):
    family, X, y = family_data()
    fitter = NewtonFitter()
    offset = jnp.zeros(y.shape[0])
    result = fitter.fit(family, X, y, offset, weights=None)
    assert isinstance(result, FitResult)


@pytest.mark.parametrize("family_data", [_gaussian_data, _poisson_data, _binomial_data])
def test_newton_fitter_converges(family_data):
    family, X, y = family_data()
    fitter = NewtonFitter()
    offset = jnp.zeros(y.shape[0])
    result = fitter.fit(family, X, y, offset, weights=None)
    assert bool(result.converged), f"Newton did not converge: objective_delta={result.objective_delta}"


@pytest.mark.parametrize("family_data", [_gaussian_data, _poisson_data, _binomial_data])
def test_newton_fitter_output_shapes(family_data):
    family, X, y = family_data()
    n, p = X.shape
    fitter = NewtonFitter()
    offset = jnp.zeros(n)
    result = fitter.fit(family, X, y, offset, weights=None)
    assert result.params.beta.shape == (p,)
    assert result.eta.shape == (n,)
    assert result.mu.shape == (n,)
    assert result.glm_wt.shape == (n,)
    assert result.score_residual.shape == (n,)


# ---------------------------------------------------------------------------
# Numerical agreement with IRLS and statsmodels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family_data", [_gaussian_data, _poisson_data, _binomial_data])
def test_newton_matches_irls(family_data):
    family, X, y = family_data()
    offset = jnp.zeros(y.shape[0])
    irls_result = glmax.IRLSFitter().fit(family, X, y, offset, weights=None)
    newton_result = NewtonFitter().fit(family, X, y, offset, weights=None)
    assert_allclose(
        np.asarray(newton_result.params.beta),
        np.asarray(irls_result.params.beta),
        rtol=1e-4,
        atol=1e-5,
        err_msg="Newton beta differs from IRLS beta",
    )


def test_newton_gaussian_matches_statsmodels():
    family, X, y = _gaussian_data()
    fitted = glmax.fit(family, X, y, fitter=NewtonFitter())
    sm_result = sm.GLM(
        np.asarray(y),
        np.asarray(X),
        family=sm.families.Gaussian(sm_links.Identity()),
    ).fit()
    assert_allclose(np.asarray(fitted.params.beta), sm_result.params, rtol=1e-4, atol=1e-5)


def test_newton_poisson_matches_statsmodels():
    family, X, y = _poisson_data()
    fitted = glmax.fit(family, X, y, fitter=NewtonFitter())
    sm_result = sm.GLM(
        np.asarray(y),
        np.asarray(X),
        family=sm.families.Poisson(sm_links.Log()),
    ).fit()
    assert_allclose(np.asarray(fitted.params.beta), sm_result.params, rtol=1e-4, atol=1e-5)


def test_newton_binomial_matches_statsmodels():
    family, X, y = _binomial_data()
    fitted = glmax.fit(family, X, y, fitter=NewtonFitter())
    sm_result = sm.GLM(
        np.asarray(y),
        np.asarray(X),
        family=sm.families.Binomial(sm_links.Logit()),
    ).fit()
    assert_allclose(np.asarray(fitted.params.beta), sm_result.params, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Offset support
# ---------------------------------------------------------------------------


def test_newton_poisson_with_offset_matches_statsmodels():
    key1, key2, key3 = jr.split(_KEY, 3)
    X = jnp.column_stack([jnp.ones(_N), jr.normal(key1, (_N,))])
    offset = 0.5 * jr.normal(key2, (_N,))
    lam = jnp.exp(X @ jnp.array([0.5, 0.4]) + offset)
    y = jr.poisson(key3, lam).astype(jnp.float64)

    fitted = glmax.fit(Poisson(LogLink()), X, y, offset=offset, fitter=NewtonFitter())
    sm_result = sm.GLM(
        np.asarray(y),
        np.asarray(X),
        family=sm.families.Poisson(sm_links.Log()),
        offset=np.asarray(offset),
    ).fit()
    assert_allclose(np.asarray(fitted.params.beta), sm_result.params, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Warm-starting
# ---------------------------------------------------------------------------


def test_newton_warm_start_converges_in_fewer_iterations():
    family, X, y = _poisson_data()
    offset = jnp.zeros(y.shape[0])
    fitter = NewtonFitter()

    cold_result = fitter.fit(family, X, y, offset, weights=None)
    warm_init = cold_result.params
    warm_result = fitter.fit(family, X, y, offset, weights=None, init=warm_init)

    # Warm start should reach the same beta
    assert_allclose(
        np.asarray(warm_result.params.beta),
        np.asarray(cold_result.params.beta),
        rtol=1e-6,
    )
    # And should take equal or fewer iterations
    assert int(warm_result.num_iters) <= int(cold_result.num_iters)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_newton_fitter_raises_on_weights():
    family, X, y = _gaussian_data()
    offset = jnp.zeros(y.shape[0])
    weights = jnp.ones(y.shape[0])
    with pytest.raises(ValueError, match="not supported"):
        NewtonFitter().fit(family, X, y, offset, weights=weights)


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


def test_newton_fitter_jit_compatible():
    family, X, y = _gaussian_data()
    fitted = eqx.filter_jit(glmax.fit)(family, X, y, fitter=NewtonFitter())
    assert isinstance(fitted, FittedGLM)
    assert bool(fitted.converged)


# ---------------------------------------------------------------------------
# Works as fitter= argument to glmax.fit
# ---------------------------------------------------------------------------


def test_newton_fitter_via_glmax_fit():
    family, X, y = _gaussian_data()
    fitted = glmax.fit(family, X, y, fitter=NewtonFitter())
    assert isinstance(fitted, FittedGLM)
    assert bool(fitted.converged)
    assert fitted.params.beta.shape == (X.shape[1],)
