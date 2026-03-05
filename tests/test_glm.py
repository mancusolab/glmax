# AbstractLinearSolver from solve
# AbstractStdErrEstimator
# FisherInfoError
# HuberError

import importlib

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
        lam = jnp.exp(eta)
        r = jnp.array(1.0 / dispersion)
        gamma_sample = rdm.gamma(noise_key, r, shape=lam.shape)
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
    assert bool(glm_state.converged)
    assert int(glm_state.num_iters) > 0


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
    assert bool(glm_state.converged)
    assert int(glm_state.num_iters) > 0


@pytest.mark.parametrize("solver", (glmax.QRSolver(), glmax.CGSolver(), glmax.CholeskySolver()))
def test_logit(getkey, solver):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate Binomial regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="binomial")

    # solve using statsmodel method (ground truth)
    sm_logit = sm.GLM(np.array(y), np.array(X), family=sm.families.Binomial())
    sm_state = sm_logit.fit()

    # solve using glmax functions
    glmax_logit = glmax.GLM(family=glmax.Binomial(), solver=solver)
    glm_state = glmax_logit.fit(X, y)

    assert_array_eq(glm_state.beta, sm_state.params, rtol=1e-3)
    assert_array_eq(glm_state.se, sm_state.bse, rtol=1e-3)
    assert_array_eq(glm_state.p, sm_state.pvalues, rtol=1e-3)
    assert bool(glm_state.converged)
    assert int(glm_state.num_iters) > 0


@pytest.mark.parametrize("solver", (glmax.QRSolver(), glmax.CGSolver(), glmax.CholeskySolver()))
def test_NegativeBinomial(getkey, solver):
    key = rdm.PRNGKey(42)  # Random seed for reproducibility
    n_samples = 200
    n_features = 5

    # Simulate NegativeBinomial regression data
    X, y, beta_true = simulate_glm_data(key, n_samples, n_features, family="negative_binomial", dispersion=2.0)

    jaxqtl_nb = glmax.GLM(family=glmax.NegativeBinomial(), solver=solver)
    glm_state = jaxqtl_nb.fit(X, y, tol=1e-8)

    # solve using statsmodel method (ground truth)
    sm_negbin = sm.GLM(np.array(y), np.array(X), family=sm.families.NegativeBinomial(alpha=glm_state.alpha))
    sm_state = sm_negbin.fit()
    sm_beta = sm_state.params
    sm_se = sm_state.bse
    sm_p = sm_state.pvalues

    assert_array_eq(glm_state.beta, sm_beta, rtol=1e-3)
    assert_array_eq(glm_state.se, sm_se, rtol=1e-3)
    assert_array_eq(glm_state.p, sm_p, rtol=1e-3)
    assert bool(glm_state.converged)
    assert int(glm_state.num_iters) > 0


def test_module_fit_entrypoint_executes(getkey):
    n_samples = 120
    n_features = 4

    X, y, _ = simulate_glm_data(getkey(), n_samples, n_features, family="poisson")
    glm_state = glmax.fit(X, y, family=glmax.Poisson(), solver=glmax.CholeskySolver())

    assert isinstance(glm_state, glmax.GLMState)
    assert glm_state.beta.shape == (n_features,)


def test_glm_fit_delegates_to_module_entrypoint(monkeypatch):
    fit_module = importlib.import_module("glmax.fit")
    model = glmax.GLM(family=glmax.Poisson(), solver=glmax.CholeskySolver())

    expected = glmax.GLMState(
        beta=jnp.array([1.0, 2.0]),
        se=jnp.array([0.1, 0.2]),
        z=jnp.array([10.0, 10.0]),
        p=jnp.array([0.0, 0.0]),
        eta=jnp.array([0.0, 0.0, 0.0]),
        mu=jnp.array([1.0, 1.0, 1.0]),
        glm_wt=jnp.array([1.0, 1.0, 1.0]),
        num_iters=jnp.asarray(2),
        converged=jnp.asarray(True),
        infor_inv=jnp.eye(2),
        resid=jnp.array([0.0, 0.0, 0.0]),
        alpha=jnp.asarray(0.0),
    )
    recorded: dict[str, object] = {}

    def fake_fit(
        X,
        y,
        family,
        solver,
        fitter=None,
        offset_eta=0.0,
        init=None,
        alpha_init=None,
        se_estimator=None,
        max_iter=1000,
        tol=1e-3,
        step_size=1.0,
    ):
        recorded["family"] = family
        recorded["solver"] = solver
        recorded["fitter"] = fitter
        recorded["offset_eta"] = offset_eta
        recorded["init"] = init
        recorded["alpha_init"] = alpha_init
        recorded["max_iter"] = max_iter
        recorded["tol"] = tol
        recorded["step_size"] = step_size
        return expected

    monkeypatch.setattr(fit_module, "fit", fake_fit)

    X = jnp.ones((3, 2))
    y = jnp.array([1.0, 2.0, 3.0])
    actual = model.fit(X, y, offset_eta=1.5, max_iter=22, tol=1e-5, step_size=0.25)

    assert actual is expected
    assert recorded["family"] is model.family
    assert recorded["solver"] is model.solver
    assert isinstance(recorded["fitter"], importlib.import_module("glmax.infer.fitters").IRLSFitter)
    assert recorded["offset_eta"] == 1.5
    assert recorded["max_iter"] == 22
    assert recorded["tol"] == 1e-5
    assert recorded["step_size"] == 0.25


def test_consolidated_infer_modules_are_available():
    contracts = importlib.import_module("glmax.infer.contracts")
    solvers = importlib.import_module("glmax.infer.solvers")
    inference = importlib.import_module("glmax.infer.inference")

    assert hasattr(contracts, "AbstractLinearSolver")
    assert hasattr(solvers, "QRSolver")
    assert hasattr(solvers, "CholeskySolver")
    assert hasattr(solvers, "CGSolver")
    assert hasattr(inference, "FisherInfoError")
    assert hasattr(inference, "HuberError")
    assert hasattr(inference, "wald_test")


def test_glm_wald_test_routes_through_inference_strategy(monkeypatch):
    glm_module = importlib.import_module("glmax.glm")
    model = glmax.GLM(family=glmax.Poisson(), solver=glmax.CholeskySolver())

    def fake_wald_test(statistic, df, family):
        return jnp.asarray([0.123456])

    monkeypatch.setattr(glm_module, "inference_wald_test", fake_wald_test)

    pval = model.wald_test(jnp.asarray([1.0]), 1)
    assert_array_eq(pval, jnp.asarray([0.123456]), atol=1e-12)


def test_infer_public_exports_remain_stable():
    from glmax.infer import (
        AbstractStdErrEstimator,
        CGSolver,
        CholeskySolver,
        FisherInfoError,
        HuberError,
        irls,
        QRSolver,
    )

    inference = importlib.import_module("glmax.infer.inference")
    solvers = importlib.import_module("glmax.infer.solvers")

    assert callable(irls)
    assert QRSolver is solvers.QRSolver
    assert CholeskySolver is solvers.CholeskySolver
    assert CGSolver is solvers.CGSolver
    assert AbstractStdErrEstimator is inference.AbstractStdErrEstimator
    assert FisherInfoError is inference.FisherInfoError
    assert HuberError is inference.HuberError


def test_legacy_infer_modules_warn_but_preserve_aliases():
    with pytest.warns(DeprecationWarning):
        legacy_solve = importlib.reload(importlib.import_module("glmax.infer.solve"))
    with pytest.warns(DeprecationWarning):
        legacy_stderr = importlib.reload(importlib.import_module("glmax.infer.stderr"))

    contracts = importlib.import_module("glmax.infer.contracts")
    solvers = importlib.import_module("glmax.infer.solvers")
    inference = importlib.import_module("glmax.infer.inference")

    assert legacy_solve.AbstractLinearSolver is contracts.AbstractLinearSolver
    assert legacy_solve.SolverState is contracts.SolverState
    assert legacy_solve.QRSolver is solvers.QRSolver
    assert legacy_solve.CholeskySolver is solvers.CholeskySolver
    assert legacy_solve.CGSolver is solvers.CGSolver
    assert legacy_stderr.AbstractStdErrEstimator is inference.AbstractStdErrEstimator
    assert legacy_stderr.FisherInfoError is inference.FisherInfoError
    assert legacy_stderr.HuberError is inference.HuberError


def test_fitters_module_provides_irls_contract():
    fitters = importlib.import_module("glmax.infer.fitters")
    optimize = importlib.import_module("glmax.infer.optimize")

    assert hasattr(fitters, "AbstractGLMFitter")
    assert hasattr(fitters, "IRLSFitter")
    assert hasattr(fitters, "IRLSState")
    assert callable(fitters.irls)
    assert optimize.irls is fitters.irls
