# AbstractLinearSolver from solve
# AbstractStdErrEstimator
# FisherInfoError
# HuberError

from typing import Tuple

import numpy as np
import pytest
import statsmodels.api as sm

from utils import assert_array_eq

import jax.nn
import jax.numpy as jnp
import jax.random as rdm

import glmax


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
    X = X - X.mean(axis=0) / (X.std(axis=0))

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
        lam = jnp.exp(eta).astype(jnp.float32)
        r = jnp.array(1.0 / dispersion, dtype=jnp.float32)
        gamma_sample = rdm.gamma(noise_key, r, shape=lam.shape, dtype=jnp.float32)
        y = rdm.poisson(extra_key, lam=gamma_sample * lam / r)  # Negative binomial regression
    else:
        raise ValueError("Unsupported family. Choose from: 'poisson', 'normal', 'binomial', 'negative_binomial'.")

    return X, y, beta_true


@pytest.mark.parametrize("solver", (glmax.QRSolver(), glmax.CGSolver(), glmax.CholeskySolver()))
def test_poisson(getkey, solver):
    n_samples = 200
    n_features = 5

    # Simulate Poisson regression data
    X, y, beta_true = simulate_glm_data(getkey(), n_samples, n_features, family="poisson")

    # solve using statsmodel method (ground truth)
    sm_poi = sm.GLM(np.array(y), np.array(X), family=sm.families.Poisson())
    sm_state = sm_poi.fit()

    # solve using glmax functions
    glmax_poi = glmax.GLM(family=glmax.Poisson(), solver=solver)
    glm_state = glmax_poi.fit(X, y)

    assert_array_eq(glm_state.beta, sm_state.params, atol=1e-3)
    assert_array_eq(glm_state.se, sm_state.bse, atol=1e-3)
    assert_array_eq(glm_state.p, sm_state.pvalues, atol=1e-3)


@pytest.mark.parametrize("solver", (glmax.QRSolver(), glmax.CGSolver(), glmax.CholeskySolver()))
def test_normal(getkey, solver):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate Normal regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="normal")

    # solve using statsmodel method (ground truth)
    sm_norm = sm.OLS(np.array(y), np.array(X))
    sm_state = sm_norm.fit()

    # solve using glmax functions
    glmax_normal = glmax.GLM(family=glmax.Gaussian(), solver=solver)
    glm_state = glmax_normal.fit(X, y)

    assert_array_eq(glm_state.beta, sm_state.params, rtol=1e-3)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-3)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-3)


@pytest.mark.parametrize("solver", (glmax.QRSolver(), glmax.CGSolver(), glmax.CholeskySolver()))
def test_logit(getkey, solver):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate Binomial regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="binomial")

    # solve using statsmodel method (ground truth)
    sm_logit = sm.Logit(np.array(y), np.array(X))
    sm_state = sm_logit.fit()

    # solve using glmax functions
    glmax_logit = glmax.GLM(family=glmax.Binomial(), solver=solver)
    glm_state = glmax_logit.fit(X, y)

    assert_array_eq(glm_state.beta, sm_state.params, rtol=1e-3)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-3)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-3)


@pytest.mark.parametrize("solver", (glmax.QRSolver(), glmax.CGSolver(), glmax.CholeskySolver()))
def test_NegativeBinomial(getkey, solver):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate NegativeBinomial regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="negative_binomial", dispersion=2.0)

    # solve using statsmodel method (ground truth)
    sm_NegativeBinomial = sm.NegativeBinomial(np.array(y), np.array(X))
    sm_state = sm_NegativeBinomial.fit()
    sm_beta = sm_state.params[:-1]
    # sm_se = sm_state.bse[:-1]
    # sm_p = sm_state.pvalues[:-1]
    sm_alpha = sm_state.params[-1]
    # sm_alpha_se = sm_state.bse[-1]

    jaxqtl_nb = glmax.GLM(family=glmax.NegativeBinomial(), solver=solver)
    glm_state = jaxqtl_nb.fit(X, y)

    print(f"iter = {glm_state.num_iters}")
    assert_array_eq(glm_state.beta, sm_beta, rtol=1e-2)
    # assert_array_eq(glm_state.se, sm_se, rtol=1e-2)
    # assert_array_eq(glm_state.p, sm_p, rtol=1e-2)
    assert_array_eq(glm_state.alpha, sm_alpha, rtol=1e-2)
