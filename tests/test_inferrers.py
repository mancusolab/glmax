# pattern: Imperative Shell

import pytest

import equinox as eqx
import jax.numpy as jnp

from jax.scipy.stats import norm

import glmax

from glmax import GLMData, InferenceResult
from glmax._infer.hyptest import AbstractTest, ScoreTest, WaldTest
from glmax._infer.infer import infer as legacy_infer, wald_test
from glmax._infer.stderr import AbstractStdErrEstimator, FisherInfoError
from glmax.family import Binomial, Gaussian, Poisson


def _make_fitted(family=None):
    if family is None:
        family = Gaussian()
    model = glmax.specify(family=family)
    data = GLMData(
        X=jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        y=jnp.array([1.2, 1.9, 3.1, 4.2]),
    )
    return glmax.fit(model, data)


def _make_poisson_fitted():
    model = glmax.specify(family=Poisson())
    data = GLMData(
        X=jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        y=jnp.array([0.0, 1.0, 1.0, 2.0]),
    )
    return glmax.fit(model, data)


def _make_binomial_fitted():
    model = glmax.specify(family=Binomial())
    data = GLMData(
        X=jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        y=jnp.array([0.0, 1.0, 1.0, 0.0]),
    )
    return glmax.fit(model, data)


def _make_perfect_fit_gaussian_fitted():
    model = glmax.specify(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        y=jnp.array([1.0, 2.0, 3.0, 4.0]),
    )
    return glmax.fit(model, data)


def _expected_score_stat(fitted):
    fit_result = fitted.result
    X = fit_result.X
    y = fit_result.y
    mu = fit_result.mu
    glm_wt = fit_result.glm_wt
    score_residual = fit_result.score_residual
    phi = jnp.asarray(fitted.model.family.scale(X, y, mu))
    numerator = X.T @ (glm_wt * score_residual)
    fisher_diag = jnp.sum(X * (glm_wt[:, jnp.newaxis] * X), axis=0)
    return numerator / jnp.sqrt(phi * fisher_diag)


def test_inferrer_types_are_strategy_objects() -> None:
    assert isinstance(WaldTest(), AbstractTest)
    assert isinstance(ScoreTest(), AbstractTest)
    assert isinstance(WaldTest(), eqx.Module)


def test_wald_inferrer_matches_legacy_infer() -> None:
    fitted = _make_fitted()

    legacy = legacy_infer(fitted)
    inferred = WaldTest()(fitted, FisherInfoError())

    assert isinstance(inferred, InferenceResult)
    assert jnp.allclose(inferred.stat, legacy.stat, atol=1e-12)
    assert jnp.allclose(inferred.se, legacy.se, atol=1e-12)
    assert jnp.allclose(inferred.p, legacy.p, atol=1e-12)


def test_wald_inferrer_gaussian_uses_t_distribution() -> None:
    fitted = _make_fitted(Gaussian())

    inferred = WaldTest()(fitted, FisherInfoError())
    expected = wald_test(inferred.stat, fitted.eta.shape[0] - fitted.params.beta.shape[0], fitted.model.family)

    assert jnp.allclose(inferred.p, expected, atol=1e-12)


def test_wald_inferrer_uses_injected_stderr() -> None:
    fitted = _make_fitted()

    class ConstantCovStdErr(AbstractStdErrEstimator):
        def __call__(self, fitted_arg):
            assert fitted_arg is fitted
            return jnp.eye(2) * 4.0

    inferred = WaldTest()(fitted, ConstantCovStdErr())

    assert jnp.allclose(inferred.se, jnp.array([2.0, 2.0]))


def test_wald_inferrer_rejects_non_fitted_glm() -> None:
    with pytest.raises(TypeError, match="FittedGLM"):
        WaldTest()(object(), FisherInfoError())


def test_score_inferrer_returns_valid_result() -> None:
    fitted = _make_fitted()

    result = ScoreTest()(fitted, FisherInfoError())

    assert isinstance(result, InferenceResult)
    assert bool(jnp.all(jnp.isfinite(result.stat)))
    assert bool(jnp.all((result.p >= 0.0) & (result.p <= 1.0)))
    assert bool(jnp.all(jnp.isnan(result.se)))


def test_score_inferrer_does_not_call_stderr() -> None:
    fitted = _make_fitted()

    class RaisingStdErr(AbstractStdErrEstimator):
        def __call__(self, fitted_arg):
            del fitted_arg
            raise RuntimeError("stderr should not be called")

    result = ScoreTest()(fitted, RaisingStdErr())

    assert isinstance(result, InferenceResult)


@pytest.mark.parametrize(
    ("make_fitted",),
    [
        pytest.param(_make_fitted, id="gaussian"),
        pytest.param(_make_poisson_fitted, id="poisson"),
    ],
)
def test_score_inferrer_matches_task_5_formula(make_fitted) -> None:
    fitted = make_fitted()

    result = ScoreTest()(fitted, FisherInfoError())
    expected_stat = _expected_score_stat(fitted)
    expected_p = 2.0 * norm.sf(jnp.abs(expected_stat))

    assert jnp.allclose(result.stat, expected_stat, atol=1e-12)
    assert jnp.allclose(result.p, expected_p, atol=1e-12)
    assert bool(jnp.all(jnp.isnan(result.se)))


@pytest.mark.parametrize(
    ("make_fitted",),
    [
        pytest.param(_make_fitted, id="gaussian"),
        pytest.param(_make_poisson_fitted, id="poisson"),
        pytest.param(_make_binomial_fitted, id="binomial"),
    ],
)
def test_score_inferrer_stat_shape_matches_beta(make_fitted) -> None:
    fitted = make_fitted()

    result = ScoreTest()(fitted, FisherInfoError())

    assert result.stat.shape == fitted.params.beta.shape


def test_score_inferrer_gaussian_p_values_are_valid() -> None:
    fitted = _make_fitted(Gaussian())

    score_result = ScoreTest()(fitted, FisherInfoError())
    expected_stat = _expected_score_stat(fitted)
    expected_p = 2.0 * norm.sf(jnp.abs(expected_stat))

    assert bool(jnp.all(jnp.isfinite(score_result.p)))
    assert bool(jnp.all((score_result.p > 0.0) & (score_result.p <= 1.0)))
    assert jnp.allclose(score_result.stat, expected_stat, atol=1e-12)
    assert jnp.allclose(score_result.p, expected_p, atol=1e-12)


def test_score_inferrer_rejects_degenerate_gaussian_scale() -> None:
    fitted = _make_perfect_fit_gaussian_fitted()

    with pytest.raises(ValueError, match="family.scale"):
        ScoreTest()(fitted, FisherInfoError())


def test_score_inferrer_rejects_degenerate_fisher_information_diagonal() -> None:
    fitted = _make_fitted()
    degenerate_X = fitted.X.at[:, 1].set(0.0)
    degenerate_fitted = eqx.tree_at(lambda tree: tree.result.X, fitted, degenerate_X)

    phi = jnp.asarray(
        degenerate_fitted.model.family.scale(
            degenerate_fitted.result.X,
            degenerate_fitted.result.y,
            degenerate_fitted.result.mu,
        )
    )
    fisher_diag = jnp.sum(
        degenerate_fitted.result.X * (degenerate_fitted.result.glm_wt[:, jnp.newaxis] * degenerate_fitted.result.X),
        axis=0,
    )

    assert bool(jnp.isfinite(phi))
    assert float(phi) > 0.0
    assert bool(jnp.any(fisher_diag <= 0.0))

    with pytest.raises(ValueError, match="Fisher information diagonal"):
        ScoreTest()(degenerate_fitted, FisherInfoError())
