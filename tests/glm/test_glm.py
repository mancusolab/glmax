# pattern: Imperative Shell

import numpy as np
import pytest
import statsmodels.api as sm

import jax.nn
import jax.numpy as jnp
import jax.random as rdm

import glmax

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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    X, y, _ = simulate_glm_data(getkey(), n_samples, n_features, family="poisson")

    # solve using statsmodel method (ground truth)
    sm_poi = sm.GLM(np.array(y), np.array(X), family=sm.families.Poisson())
    sm_state = sm_poi.fit()

    # solve using glmax functions
    glm_state = glmax.fit(Poisson(), X, y)
    infer_state = glmax.infer(glm_state)

    _assert_array_eq(glm_state.params.beta, sm_state.params, atol=1e-3)
    _assert_array_eq(infer_state.se, sm_state.bse, atol=1e-3)
    _assert_array_eq(infer_state.p, sm_state.pvalues, atol=1e-3)


def test_normal(getkey):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate Normal regression data
    X, y, _ = simulate_glm_data(key, n_samples, n_features, family="normal")

    # solve using statsmodel method (ground truth)
    sm_norm = sm.OLS(np.array(y), np.array(X))
    sm_state = sm_norm.fit()

    # solve using glmax functions
    glm_state = glmax.fit(Gaussian(), X, y)
    infer_state = glmax.infer(glm_state)

    _assert_array_eq(glm_state.params.beta, sm_state.params, rtol=1e-3)
    _assert_array_eq(infer_state.se, sm_state.bse, rtol=1e-3)
    _assert_array_eq(infer_state.p, sm_state.pvalues, rtol=1e-3)


def test_logit(getkey):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate Binomial regression data
    X, y, _ = simulate_glm_data(key, n_samples, n_features, family="binomial")

    # solve using statsmodel method (ground truth)
    sm_logit = sm.GLM(np.array(y), np.array(X), family=sm.families.Binomial())
    sm_state = sm_logit.fit()

    # solve using glmax functions
    glm_state = glmax.fit(Binomial(), X, y)
    infer_state = glmax.infer(glm_state)

    _assert_array_eq(glm_state.params.beta, sm_state.params, rtol=1e-3)
    _assert_array_eq(infer_state.se, sm_state.bse, rtol=1e-3)
    _assert_array_eq(infer_state.p, sm_state.pvalues, rtol=1e-3)


def test_NegativeBinomial(getkey):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate NegativeBinomial regression data
    X, y, _ = simulate_glm_data(key, n_samples, n_features, family="negative_binomial", dispersion=2.0)

    glm_state = glmax.fit(NegativeBinomial(), X, y)
    infer_state = glmax.infer(glm_state)
    assert glm_state.params._fields == ("beta", "disp", "aux")
    assert jnp.allclose(glm_state.params.disp, jnp.array(1.0))
    assert glm_state.params.aux is not None
    assert float(jnp.asarray(glm_state.params.aux)) > 0.0

    # solve using statsmodel method (ground truth)
    sm_negbin = sm.GLM(np.array(y), np.array(X), family=sm.families.NegativeBinomial(alpha=glm_state.params.aux))
    sm_state = sm_negbin.fit()
    sm_beta = sm_state.params
    sm_se = sm_state.bse
    sm_p = sm_state.pvalues

    _assert_array_eq(glm_state.params.beta, sm_beta, rtol=6e-3)
    _assert_array_eq(infer_state.se, sm_se, rtol=5e-3)
    _assert_array_eq(infer_state.p, sm_p, rtol=4e-2)


@pytest.mark.parametrize("n_trials", [5, 10, 25])
def test_binomial_n_trials_matches_statsmodels(n_trials):
    import jax.random as jr

    key = jr.PRNGKey(42 + n_trials)
    key_X, key_y = jr.split(key)
    n = 100
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key_X, (n, 2))], axis=1)
    true_eta = X @ jnp.array([0.2, 0.5, -0.3])
    prob = jax.nn.sigmoid(true_eta)
    counts = jr.binomial(key_y, n=n_trials, p=prob, shape=(n,)).astype(jnp.float64)

    glmax_result = glmax.fit(Binomial(n_trials=n_trials), X, counts)

    # statsmodels expects (successes, failures) for grouped binomial
    endog = np.column_stack([np.array(counts), np.full(n, n_trials) - np.array(counts)])
    sm_result = sm.GLM(endog, np.array(X), family=sm.families.Binomial()).fit()

    _assert_array_eq(glmax_result.params.beta, jnp.array(sm_result.params), rtol=1e-3)
