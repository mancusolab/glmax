# pattern: Imperative Shell

import pytest

import equinox as eqx
import jax.numpy as jnp

from jax.scipy.stats import norm

import glmax

from glmax import InferenceResult
from glmax._infer.hyptest import _wald_test, AbstractTest, ScoreTest, WaldTest
from glmax._infer.infer import infer as legacy_infer
from glmax._infer.stderr import AbstractStdErrEstimator, FisherInfoError
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


def _make_fitted(family=None):
    if family is None:
        family = Gaussian()
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([1.2, 1.9, 3.1, 4.2])
    return glmax.fit(family, X, y)


def _make_poisson_fitted():
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([0.0, 1.0, 1.0, 2.0])
    return glmax.fit(Poisson(), X, y)


def _make_binomial_fitted():
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([0.0, 1.0, 1.0, 0.0])
    return glmax.fit(Binomial(), X, y)


def _make_negative_binomial_fitted():
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([0.0, 1.0, 2.0, 4.0])
    return glmax.fit(NegativeBinomial(), X, y)


def _make_perfect_fit_gaussian_fitted():
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = jnp.array([1.0, 2.0, 3.0, 4.0])
    return glmax.fit(Gaussian(), X, y)


def _expected_score_stat(fitted):
    fit_result = fitted.result
    X = fit_result.X
    glm_wt = fit_result.glm_wt
    score_residual = fit_result.score_residual
    phi = jnp.asarray(fitted.params.disp)
    numerator = X.T @ (glm_wt * score_residual)
    fisher_diag = jnp.sum(X * (glm_wt[:, jnp.newaxis] * X), axis=0)
    return numerator / jnp.sqrt(phi * fisher_diag)


def _with_dispersion(fitted, disp):
    params = fitted.params._replace(disp=jnp.asarray(disp))
    return eqx.tree_at(lambda tree: tree.result.params, fitted, params)


def test_hypothesis_tests_are_strategy_objects() -> None:
    assert isinstance(WaldTest(), AbstractTest)
    assert isinstance(ScoreTest(), AbstractTest)
    assert isinstance(WaldTest(), eqx.Module)


def test_wald_test_matches_legacy_infer() -> None:
    fitted = _make_fitted()

    legacy = legacy_infer(fitted)
    inferred = WaldTest()(fitted, FisherInfoError())

    assert isinstance(inferred, InferenceResult)
    assert jnp.allclose(inferred.stat, legacy.stat, atol=1e-12)
    assert jnp.allclose(inferred.se, legacy.se, atol=1e-12)
    assert jnp.allclose(inferred.p, legacy.p, atol=1e-12)


def test_wald_test_uses_injected_stderr() -> None:
    fitted = _make_fitted()

    class ConstantCovStdErr(AbstractStdErrEstimator):
        def __call__(self, fitted_arg):
            assert fitted_arg is fitted
            return jnp.eye(2) * 4.0

    inferred = WaldTest()(fitted, ConstantCovStdErr())

    assert jnp.allclose(inferred.se, jnp.array([2.0, 2.0]))


def test_wald_test_rejects_non_fitted_glm() -> None:
    with pytest.raises(TypeError, match="FittedGLM"):
        WaldTest()(object(), FisherInfoError())


def test_score_test_returns_valid_result() -> None:
    fitted = _make_fitted()

    result = ScoreTest()(fitted, FisherInfoError())

    assert isinstance(result, InferenceResult)
    assert bool(jnp.all(jnp.isfinite(result.stat)))
    assert bool(jnp.all((result.p >= 0.0) & (result.p <= 1.0)))
    assert bool(jnp.all(jnp.isnan(result.se)))


def test_score_test_does_not_call_stderr() -> None:
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
def test_score_test_matches_task_formula(make_fitted) -> None:
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
def test_score_test_stat_shape_matches_beta(make_fitted) -> None:
    fitted = make_fitted()

    result = ScoreTest()(fitted, FisherInfoError())

    assert result.stat.shape == fitted.params.beta.shape


def test_score_test_gaussian_p_values_valid() -> None:
    fitted = _make_fitted(Gaussian())

    score_result = ScoreTest()(fitted, FisherInfoError())
    expected_stat = _expected_score_stat(fitted)
    expected_p = 2.0 * norm.sf(jnp.abs(expected_stat))

    assert bool(jnp.all(jnp.isfinite(score_result.p)))
    assert bool(jnp.all((score_result.p > 0.0) & (score_result.p <= 1.0)))
    assert jnp.allclose(score_result.stat, expected_stat, atol=1e-12)
    assert jnp.allclose(score_result.p, expected_p, atol=1e-12)


def test_score_test_rejects_degenerate_scale() -> None:
    fitted = _make_perfect_fit_gaussian_fitted()

    with pytest.raises(ValueError, match="finite and > 0"):
        ScoreTest()(fitted, FisherInfoError())


def test_score_test_rejects_nonpositive_fitted_dispersion() -> None:
    fitted = _with_dispersion(_make_fitted(), 0.0)

    with pytest.raises(ValueError, match="fitted.params.disp"):
        ScoreTest()(fitted, FisherInfoError())


@pytest.mark.parametrize(
    ("make_fitted", "disp"),
    [
        pytest.param(_make_fitted, 4.0, id="gaussian"),
        pytest.param(_make_negative_binomial_fitted, 3.0, id="negative-binomial"),
    ],
)
def test_score_test_uses_fitted_dispersion_as_phi_source_of_truth(make_fitted, disp) -> None:
    fitted = _with_dispersion(make_fitted(), disp)

    result = ScoreTest()(fitted, FisherInfoError())
    expected_stat = _expected_score_stat(fitted)
    expected_p = 2.0 * norm.sf(jnp.abs(expected_stat))

    assert jnp.allclose(result.stat, expected_stat, atol=1e-12)
    assert jnp.allclose(result.p, expected_p, atol=1e-12)


def test_score_test_rejects_degenerate_fisher_diag() -> None:
    fitted = _make_fitted()
    degenerate_X = fitted.X.at[:, 1].set(0.0)
    degenerate_fitted = eqx.tree_at(lambda tree: tree.result.X, fitted, degenerate_X)

    phi = jnp.asarray(degenerate_fitted.params.disp)
    fisher_diag = jnp.sum(
        degenerate_fitted.result.X * (degenerate_fitted.result.glm_wt[:, jnp.newaxis] * degenerate_fitted.result.X),
        axis=0,
    )

    assert bool(jnp.isfinite(phi))
    assert float(phi) > 0.0
    assert bool(jnp.any(fisher_diag <= 0.0))

    with pytest.raises(ValueError, match="Fisher information diagonal"):
        ScoreTest()(degenerate_fitted, FisherInfoError())


def test_wald_test_uses_t_distribution_for_gaussian() -> None:
    statistic = jnp.array([2.0, -1.5, 0.0])
    p = _wald_test(statistic, df=50, family=Gaussian())

    assert p.shape == (3,)
    assert bool(jnp.all(p > 0))
    assert bool(jnp.all(p <= 1.0))
    # z=0 → p=1
    assert bool(jnp.isclose(p[2], 1.0, atol=1e-5))


def test_wald_test_uses_normal_for_non_gaussian() -> None:
    statistic = jnp.array([1.96])
    p_poisson = _wald_test(statistic, df=100, family=Poisson())
    # 2 * norm.sf(1.96) ≈ 0.05
    assert bool(jnp.abs(p_poisson[0] - 0.05) < 0.005)


@pytest.mark.parametrize("family", [Gaussian(), Poisson()])
def test_wald_test_jit_safe(family):
    """_wald_test is JIT-safe for Gaussian and Poisson."""
    stat = jnp.array([2.0, -1.5, 0.5])

    import equinox as eqx

    def _jit_wald(stat):
        return _wald_test(stat, 50, family)

    p_jit = eqx.filter_jit(_jit_wald)(stat)

    assert bool(jnp.all(jnp.isfinite(p_jit))), (
        f"JIT wald_test p-values must be finite for {type(family).__name__}; got {p_jit}"
    )
