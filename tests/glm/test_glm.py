# pattern: Imperative Shell

from typing import ClassVar, Tuple

import numpy as np
import pytest
import statsmodels.api as sm

import jax.nn
import jax.numpy as jnp
import jax.random as rdm

import glmax

from glmax import GLMData
from glmax.family import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson
from glmax.family.dist import ExponentialDispersionFamily
from glmax.family.links import IdentityLink


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


class _AuxiliaryWarmStartFamily(ExponentialDispersionFamily):
    glink: IdentityLink = IdentityLink()
    _links: ClassVar[list[type[IdentityLink]]] = [IdentityLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def scale(self, X, y, mu):
        del X, y, mu
        return jnp.asarray(1.0)

    def negloglikelihood(self, y, eta, disp=0.0, aux=None):
        del disp, aux
        return jnp.sum(jnp.square(jnp.asarray(y) - jnp.asarray(eta)))

    def variance(self, mu, disp=0.0, aux=None):
        del disp, aux
        return jnp.ones_like(jnp.asarray(mu))

    def sample(self, key, eta, disp=0.0, aux=None):
        del key, disp, aux
        return jnp.asarray(eta)

    def update_dispersion(self, X, y, eta, disp=0.0, step_size=1.0, aux=None):
        del X, y, eta, step_size, aux
        return jnp.asarray(disp) + 2.0

    def estimate_dispersion(
        self, X, y, eta, disp=0.0, step_size=1.0, aux=None, tol=1e-3, max_iter=1000, offset_eta=0.0
    ):
        del X, y, eta, step_size, aux, tol, max_iter, offset_eta
        return jnp.asarray(disp) + 3.0

    def canonical_dispersion(self, disp=0.0):
        return jnp.asarray(disp) + 1.0

    def canonical_auxiliary(self, aux=None):
        if aux is None:
            return jnp.asarray(0.25)
        return jnp.asarray(aux) + 0.5


class _LegacyCalcWeightFamily(ExponentialDispersionFamily):
    glink: IdentityLink = IdentityLink()
    _links: ClassVar[list[type[IdentityLink]]] = [IdentityLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def scale(self, X, y, mu):
        del X, y, mu
        return jnp.asarray(1.0)

    def negloglikelihood(self, y, eta, disp=0.0, aux=None):
        del disp, aux
        return jnp.sum(jnp.square(jnp.asarray(y) - jnp.asarray(eta)))

    def variance(self, mu, disp=0.0, aux=None):
        del aux
        return jnp.ones_like(jnp.asarray(mu)) * (jnp.asarray(disp) + 1.0)

    def sample(self, key, eta, disp=0.0, aux=None):
        del key, disp, aux
        return jnp.asarray(eta)

    def calc_weight(self, eta, disp=0.0, aux=None):
        mu = jnp.asarray(eta)
        aux_shift = jnp.asarray(0.0 if aux is None else aux)
        variance = jnp.ones_like(mu) * (jnp.asarray(disp) + 2.0 + aux_shift)
        weight = jnp.ones_like(mu) * (7.0 + aux_shift)
        return mu, variance, weight


class _MissingAuxLogProbFamily(ExponentialDispersionFamily):
    glink: IdentityLink = IdentityLink()
    _links: ClassVar[list[type[IdentityLink]]] = [IdentityLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def scale(self, X, y, mu):
        del X, y, mu
        return jnp.asarray(1.0)

    def negloglikelihood(self, y, eta, disp=0.0):
        del disp
        return jnp.sum(jnp.square(jnp.asarray(y) - jnp.asarray(eta)))

    def variance(self, mu, disp=0.0, aux=None):
        del disp, aux
        return jnp.ones_like(jnp.asarray(mu))

    def sample(self, key, eta, disp=0.0, aux=None):
        del key, disp, aux
        return jnp.asarray(eta)


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
    """GLM.working_weights returns (mu, g', w) tuple of correct shapes."""
    model = glmax.GLM(family=Gaussian())
    eta = jnp.array([0.5, 1.0, 1.5, 2.0])
    disp = 1.0

    result = model.working_weights(eta, disp)

    assert len(result) == 3
    mu, g_deriv, w = result
    assert mu.shape == eta.shape
    assert g_deriv.shape == eta.shape
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


def test_glm_negative_binomial_log_prob_forwards_aux() -> None:
    model = glmax.GLM(family=NegativeBinomial())
    y = jnp.array([0.0, 2.0, 5.0])
    eta = jnp.log(jnp.array([1.5, 2.5, 4.0]))

    result = model.log_prob(y, eta, disp=jnp.array(1.0), aux=jnp.array(0.4))
    expected = -model.family.negloglikelihood(y, eta, disp=jnp.array(1.0), aux=jnp.array(0.4))

    assert jnp.allclose(result, expected)
    assert jnp.allclose(result, model.log_prob(y, eta, disp=jnp.array(9.0), aux=jnp.array(0.4)))
    assert not jnp.allclose(result, model.log_prob(y, eta, disp=jnp.array(1.0), aux=jnp.array(0.2)))


def test_glm_negative_binomial_working_weights_forward_aux() -> None:
    model = glmax.GLM(family=NegativeBinomial())
    eta = jnp.log(jnp.array([1.5, 2.5, 4.0]))

    result = model.working_weights(eta, disp=jnp.array(1.0), aux=jnp.array(0.4))
    expected_mu, _, expected_weight = model.family.calc_weight(eta, disp=jnp.array(1.0), aux=jnp.array(0.4))
    expected = (expected_mu, model.link_deriv(expected_mu), expected_weight)
    ignored_disp = model.working_weights(eta, disp=jnp.array(9.0), aux=jnp.array(0.4))
    changed_aux = model.working_weights(eta, disp=jnp.array(1.0), aux=jnp.array(0.2))

    for actual, truth in zip(result, expected, strict=True):
        assert jnp.allclose(actual, truth)
    for actual, truth in zip(result, ignored_disp, strict=True):
        assert jnp.allclose(actual, truth)
    assert any(not jnp.allclose(actual, changed) for actual, changed in zip(result, changed_aux, strict=True))


def test_glm_working_weights_preserves_custom_calc_weight_override_with_aux() -> None:
    model = glmax.GLM(family=_LegacyCalcWeightFamily())
    eta = jnp.array([0.2, 0.5, 0.8])
    disp = jnp.array(1.5)
    aux = jnp.array(0.4)

    result = model.working_weights(eta, disp=disp, aux=aux)
    expected_mu, _, expected_weight = model.family.calc_weight(eta, disp=disp, aux=aux)
    expected = (expected_mu, model.link_deriv(expected_mu), expected_weight)

    for actual, truth in zip(result, expected, strict=True):
        assert jnp.allclose(actual, truth)


def test_glm_negative_binomial_sample_forwards_aux() -> None:
    key = rdm.PRNGKey(7)
    model = glmax.GLM(family=NegativeBinomial())
    eta = jnp.log(jnp.array([1.5, 2.5, 4.0]))

    result = model.sample(key, eta, disp=jnp.array(1.0), aux=jnp.array(0.4))
    expected = model.family.sample(key, eta, disp=jnp.array(1.0), aux=jnp.array(0.4))
    ignored_disp = model.sample(key, eta, disp=jnp.array(9.0), aux=jnp.array(0.4))
    changed_aux = model.sample(key, eta, disp=jnp.array(1.0), aux=jnp.array(0.2))

    assert jnp.array_equal(result, expected)
    assert jnp.array_equal(result, ignored_disp)
    assert not jnp.array_equal(result, changed_aux)


def test_glm_requires_aux_aware_family_signatures_when_aux_is_passed() -> None:
    model = glmax.GLM(family=_MissingAuxLogProbFamily())
    y = jnp.array([0.0, 1.0, 2.0])
    eta = jnp.array([0.2, 0.5, 0.8])

    with pytest.raises(TypeError, match="aux"):
        model.log_prob(y, eta, disp=jnp.array(1.5), aux=jnp.array(0.4))


def test_glm_forwards_aux_to_family_methods() -> None:
    model = glmax.GLM(family=_AuxiliaryWarmStartFamily())
    X = jnp.array([[1.0], [1.0], [1.0]])
    y = jnp.array([0.0, 1.0, 2.0])
    eta = jnp.array([0.2, 0.5, 0.8])
    disp = jnp.array(1.5)
    aux = jnp.array(0.4)
    step_size = jnp.array(0.7)
    key = rdm.PRNGKey(5)

    log_prob = model.log_prob(y, eta, disp=disp, aux=aux)
    sample = model.sample(key, eta, disp=disp, aux=aux)
    mu, g_deriv, weight = model.working_weights(eta, disp=disp, aux=aux)
    updated_disp = model.update_dispersion(X, y, eta, disp=disp, step_size=step_size, aux=aux)
    estimated_disp = model.estimate_dispersion(X, y, eta, disp=disp, aux=aux)

    expected_log_prob = -model.family.negloglikelihood(y, eta, disp=disp, aux=aux)
    expected_sample = model.family.sample(key, eta, disp=disp, aux=aux)
    expected_mu = model.mean(eta)
    expected_variance = model.family.variance(expected_mu, disp=disp, aux=aux)
    expected_g_deriv = model.link_deriv(expected_mu)
    expected_weight = 1.0 / (expected_variance * expected_g_deriv**2)
    expected_updated_disp = model.family.update_dispersion(X, y, eta, disp=disp, step_size=step_size, aux=aux)
    expected_estimated_disp = model.family.estimate_dispersion(X, y, eta, disp=disp, aux=aux)

    assert jnp.allclose(log_prob, expected_log_prob)
    assert jnp.allclose(sample, expected_sample)
    assert jnp.allclose(mu, expected_mu)
    assert jnp.allclose(g_deriv, expected_g_deriv)
    assert jnp.allclose(weight, expected_weight)
    assert jnp.allclose(updated_disp, expected_updated_disp)
    assert jnp.allclose(estimated_disp, expected_estimated_disp)


def test_glm_docstrings_describe_split_disp_aux_contract() -> None:
    assert glmax.GLM.__doc__ is not None
    assert "(disp, aux)" in glmax.GLM.__doc__

    for method in (
        glmax.GLM.log_prob,
        glmax.GLM.sample,
        glmax.GLM.working_weights,
        glmax.GLM.canonicalize_auxiliary,
        glmax.GLM.canonicalize_params,
    ):
        assert method.__doc__ is not None
        assert "`disp`" in method.__doc__
        assert "`aux`" in method.__doc__


def test_glm_canonicalize_auxiliary_ignores_aux_for_gaussian() -> None:
    model = glmax.GLM(family=Gaussian())

    assert model.canonicalize_auxiliary(jnp.array(0.2)) is None


@pytest.mark.parametrize("family", [Poisson(), Binomial(), Gamma()])
def test_glm_canonicalize_auxiliary_ignores_aux_for_families_without_aux_state(family) -> None:
    model = glmax.GLM(family=family)

    assert model.canonicalize_auxiliary(jnp.array(0.2)) is None


@pytest.mark.parametrize(
    ("family", "y", "eta"),
    [
        (Poisson(), jnp.array([0.0, 1.0, 2.0]), jnp.log(jnp.array([1.2, 2.0, 3.5]))),
        (Binomial(), jnp.array([0.0, 1.0, 1.0]), jnp.array([-0.8, 0.3, 1.1])),
    ],
)
def test_glm_methods_ignore_aux_for_families_without_aux_state(family, y, eta) -> None:
    model = glmax.GLM(family=family)
    key = rdm.PRNGKey(9)
    baseline_log_prob = model.log_prob(y, eta, disp=jnp.array(1.0), aux=None)
    with_aux_log_prob = model.log_prob(y, eta, disp=jnp.array(1.0), aux=jnp.array(0.2))
    baseline_weight = model.working_weights(eta, disp=jnp.array(1.0), aux=None)
    with_aux_weight = model.working_weights(eta, disp=jnp.array(1.0), aux=jnp.array(0.2))
    baseline_sample = model.sample(key, eta, disp=jnp.array(1.0), aux=None)
    with_aux_sample = model.sample(key, eta, disp=jnp.array(1.0), aux=jnp.array(0.2))

    assert jnp.allclose(with_aux_log_prob, baseline_log_prob)
    for actual, truth in zip(with_aux_weight, baseline_weight, strict=True):
        assert jnp.allclose(actual, truth)
    assert jnp.array_equal(with_aux_sample, baseline_sample)


def test_glm_canonicalize_params_routes_negative_binomial_alpha_to_aux() -> None:
    model = glmax.GLM(family=NegativeBinomial())

    canonical_disp, canonical_aux = model.canonicalize_params(jnp.array(3.5), jnp.array(0.2))

    assert jnp.allclose(canonical_disp, jnp.array(1.0))
    assert canonical_aux is not None
    assert jnp.allclose(canonical_aux, jnp.array(0.2))


def test_glm_canonicalize_params_delegates_warm_start_values_through_model_boundary() -> None:
    model = glmax.GLM(family=_AuxiliaryWarmStartFamily())

    canonical_disp, canonical_aux = model.canonicalize_params(jnp.array(0.5), None)

    assert jnp.allclose(canonical_disp, jnp.array(1.5))
    assert jnp.allclose(canonical_aux, jnp.array(0.25))
