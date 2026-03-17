# pattern: Imperative Shell

from typing import Tuple

import numpy as np
import statsmodels.api as sm

import jax.nn
import jax.numpy as jnp
import jax.random as rdm

import glmax

from glmax import GLMData
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


# ---------------------------------------------------------------------------
# Shared helper (replaces utils.py assert_array_eq)
# ---------------------------------------------------------------------------


def _assert_array_eq(estimate, truth, rtol=1e-7, atol=1e-8):
    import numpy.testing as nptest

    nptest.assert_allclose(estimate, truth, rtol=rtol, atol=atol)


def simulate_glm_data(
    key: rdm.PRNGKey,
    n_samples: int = 100,
    n_features: int = 5,
    family: str = "poisson",
    dispersion: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulates Generalized Linear Model (GLM) data.

    Parameters:
    -----------
    key : PRNGKey
        Random number generator key for JAX.
    n_samples : int
        Number of observations.
    n_features : int
        Number of predictor variables.
    family : str
        Specifies the exponential family ("poisson", "normal", etc.)

    Returns:
    --------
    X : jnp.ndarray
        Covariate matrix of shape (n_samples, n_features).
    y : jnp.ndarray
        Simulated response vector.
    beta_true : jnp.ndarray
        True coefficient values used for data generation.
    """
    key, x_key, beta_key, noise_key, extra_key = rdm.split(key, 5)

    # Generate random design matrix X
    X = rdm.normal(x_key, shape=(n_samples, n_features))
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Generate true coefficients
    beta_true = rdm.normal(beta_key, shape=(n_features,))

    # Compute linear predictor
    eta = X @ beta_true

    # Simulate response variable y based on family
    if family == "poisson":
        rate = jnp.exp(eta)
        y = rdm.poisson(noise_key, rate)  # Poisson regression
    elif family == "normal":
        y = eta + rdm.normal(noise_key, shape=(n_samples,))  # Normal regression
    elif family == "binomial":
        p = jnp.clip(jax.nn.sigmoid(eta), 1e-5, 1 - 1e-5)
        y = rdm.bernoulli(noise_key, p).astype(jnp.int32)  # Binomial regression
    elif family == "negative_binomial":
        lam = jnp.exp(eta)
        r = jnp.array(1.0 / dispersion)
        gamma_sample = rdm.gamma(noise_key, r, shape=lam.shape)
        y = rdm.poisson(extra_key, lam=gamma_sample * lam / r)  # Negative binomial regression
    else:
        raise ValueError("Unsupported family. Choose from: 'poisson', 'normal', 'binomial', 'negative_binomial'.")

    return X, y, beta_true


def test_poisson(getkey):
    n_samples = 200
    n_features = 5

    # Simulate Poisson regression data
    X, y, beta_true = simulate_glm_data(getkey(), n_samples, n_features, family="poisson")

    # solve using statsmodel method (ground truth)
    sm_poi = sm.GLM(np.array(y), np.array(X), family=sm.families.Poisson())
    sm_state = sm_poi.fit()

    # solve using glmax functions
    glmax_poi = glmax.specify(family=Poisson())
    glm_state = glmax.fit(glmax_poi, GLMData(X=X, y=y))
    infer_state = glmax.infer(glm_state)

    _assert_array_eq(glm_state.params.beta, sm_state.params, atol=1e-3)
    _assert_array_eq(infer_state.se, sm_state.bse, atol=1e-3)
    _assert_array_eq(infer_state.p, sm_state.pvalues, atol=1e-3)


def test_normal(getkey):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate Normal regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="normal")

    # solve using statsmodel method (ground truth)
    sm_norm = sm.OLS(np.array(y), np.array(X))
    sm_state = sm_norm.fit()

    # solve using glmax functions
    glmax_normal = glmax.specify(family=Gaussian())
    glm_state = glmax.fit(glmax_normal, GLMData(X=X, y=y))
    infer_state = glmax.infer(glm_state)

    _assert_array_eq(glm_state.params.beta, sm_state.params, rtol=1e-3)
    _assert_array_eq(infer_state.se, sm_state.bse, rtol=1e-3)
    _assert_array_eq(infer_state.p, sm_state.pvalues, rtol=1e-3)


def test_logit(getkey):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate Binomial regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="binomial")

    # solve using statsmodel method (ground truth)
    sm_logit = sm.GLM(np.array(y), np.array(X), family=sm.families.Binomial())
    sm_state = sm_logit.fit()

    # solve using glmax functions
    glmax_logit = glmax.specify(family=Binomial())
    glm_state = glmax.fit(glmax_logit, GLMData(X=X, y=y))
    infer_state = glmax.infer(glm_state)

    _assert_array_eq(glm_state.params.beta, sm_state.params, rtol=1e-3)
    _assert_array_eq(infer_state.se, sm_state.bse, rtol=1e-3)
    _assert_array_eq(infer_state.p, sm_state.pvalues, rtol=1e-3)


def test_NegativeBinomial(getkey):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate NegativeBinomial regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="negative_binomial", dispersion=2.0)

    jaxqtl_nb = glmax.specify(family=NegativeBinomial())
    glm_state = glmax.fit(jaxqtl_nb, GLMData(X=X, y=y))
    infer_state = glmax.infer(glm_state)

    # solve using statsmodel method (ground truth)
    sm_negbin = sm.GLM(np.array(y), np.array(X), family=sm.families.NegativeBinomial(alpha=glm_state.params.disp))
    sm_state = sm_negbin.fit()
    sm_beta = sm_state.params
    sm_se = sm_state.bse
    sm_p = sm_state.pvalues

    _assert_array_eq(glm_state.params.beta, sm_beta, rtol=6e-3)
    _assert_array_eq(infer_state.se, sm_se, rtol=5e-3)
    _assert_array_eq(infer_state.p, sm_p, rtol=4e-2)


# ---------------------------------------------------------------------------
# New GLM-method unit tests
# ---------------------------------------------------------------------------


def test_glm_mean_delegates_to_family_link_inverse() -> None:
    """GLM.mean(eta) equals IdentityLink inverse for Gaussian."""
    from glmax.family.links import IdentityLink

    model = glmax.GLM(family=Gaussian())
    eta = jnp.array([0.0, 1.0])
    result = model.mean(eta)
    expected = IdentityLink().inverse(eta)
    assert jnp.allclose(result, expected)


def test_glm_log_prob_is_negative_negloglikelihood() -> None:
    """GLM.log_prob = -negloglikelihood."""
    model = glmax.GLM(family=Gaussian())
    y = jnp.array([1.0, 2.0, 3.0])
    eta = jnp.array([1.1, 1.9, 3.1])
    disp = 0.5

    log_prob = model.log_prob(y, eta, disp)
    nll = model.family.negloglikelihood(y, eta, disp)

    assert jnp.allclose(log_prob, -nll)


def test_glm_working_weights_returns_triple() -> None:
    """GLM.working_weights returns (mu, v, w) tuple of correct shapes."""
    model = glmax.GLM(family=Gaussian())
    eta = jnp.array([0.5, 1.0, 1.5, 2.0])
    disp = 1.0

    result = model.working_weights(eta, disp)

    assert len(result) == 3
    mu, v, w = result
    assert mu.shape == eta.shape
    assert v.shape == eta.shape
    assert w.shape == eta.shape


def test_glm_link_deriv_matches_family() -> None:
    """GLM.link_deriv(mu) matches family.glink.deriv(mu)."""
    model = glmax.GLM(family=Gaussian())
    mu = jnp.array([0.5, 1.0, 2.0])

    result = model.link_deriv(mu)
    expected = model.family.glink.deriv(mu)

    assert jnp.allclose(result, expected)


def test_glm_scale_delegates() -> None:
    """GLM.scale(X, y, mu) matches family.scale(X, y, mu)."""
    model = glmax.GLM(family=Gaussian())
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = jnp.array([1.0, 2.0, 3.0])
    mu = jnp.array([1.1, 1.9, 3.1])

    result = model.scale(X, y, mu)
    expected = model.family.scale(X, y, mu)

    assert jnp.allclose(result, expected)


def test_glm_init_eta_matches_family() -> None:
    """GLM.init_eta(y) matches family.init_eta(y)."""
    model = glmax.GLM(family=Gaussian())
    y = jnp.array([1.0, 2.0, 3.0, 4.0])

    result = model.init_eta(y)
    expected = model.family.init_eta(y)

    assert jnp.allclose(result, expected)


def test_glm_canonicalize_dispersion_matches_family() -> None:
    """GLM.canonicalize_dispersion(disp) matches family.canonical_dispersion(disp)."""
    model = glmax.GLM(family=Gaussian())
    disp = 2.5

    result = model.canonicalize_dispersion(disp)
    expected = model.family.canonical_dispersion(disp)

    assert jnp.allclose(result, expected)


def test_glm_sample_delegates() -> None:
    """GLM.sample(key, eta, disp) matches family.sample(key, eta, disp)."""
    import jax.random as jr

    model = glmax.GLM(family=Gaussian())
    key = jr.PRNGKey(42)
    eta = jnp.array([0.0, 1.0, 2.0])
    disp = 1.0

    result = model.sample(key, eta, disp)
    expected = model.family.sample(key, eta, disp)

    assert jnp.allclose(result, expected)
