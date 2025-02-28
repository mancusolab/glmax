# let's test functions
# AbstractLinearSolver from solve
# QRSolver
# CholeskySolver from solver
# CGSolver
# AbstractStdErrEstimator
# FisherInfoError
# HuberError

from typing import Tuple

import numpy as np

from statsmodels.discrete.discrete_model import Poisson as smPoisson
from utils import assert_array_eq

import jax.numpy as jnp
import jax.random as rdm

import glmax


def simulate_glm_data(
    key: rdm.PRNGKey, n_samples: int = 100, n_features: int = 5, family: str = "poisson"
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
    key, x_key, beta_key, noise_key = rdm.split(key, 4)

    # Generate random design matrix X
    X = rdm.normal(x_key, shape=(n_samples, n_features))

    # Generate true coefficients
    beta_true = rdm.normal(beta_key, shape=(n_features,))

    # Compute linear predictor
    eta = X @ beta_true

    # Simulate response variable y based on family
    if family == "poisson":
        y = rdm.poisson(noise_key, jnp.exp(eta))  # Poisson regression
    elif family == "normal":
        y = eta + rdm.normal(noise_key, shape=(n_samples,))  # Normal regression
    else:
        raise ValueError("Unsupported family: choose 'poisson' or 'normal'.")

    return X, y, beta_true


def test_poisson_cho():
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate Poisson regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="poisson")

    mod = smPoisson(np.array(y), np.array(X))
    sm_state = mod.fit()

    glmax_poisson_cho = glmax.GLM(family=glmax.Poisson(), solver=glmax.CholeskySolver())
    init_pois = glmax_poisson_cho.family.init_eta(y.reshape(-1, 1))
    glm_state = glmax_poisson_cho.fit(X, y.reshape(-1, 1), init=init_pois)

    assert_array_eq(glm_state.beta, sm_state.params)
    assert_array_eq(glm_state.se, sm_state.bse)
    assert_array_eq(glm_state.p, sm_state.pvalues)
