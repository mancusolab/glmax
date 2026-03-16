# pattern: Imperative Shell

import importlib

import pytest

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import glmax

from glmax import AbstractFitter, FitResult, GLMData, Params
from glmax._fit.solve import AbstractLinearSolver, CholeskySolver
from glmax.family import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson


def _make_fit_result() -> FitResult:
    return FitResult(
        params=Params(beta=jnp.array([1.0]), disp=jnp.array(0.0)),
        X=jnp.array([[1.0], [1.0]]),
        y=jnp.array([1.0, 1.0]),
        eta=jnp.array([1.0, 1.0]),
        mu=jnp.array([1.0, 1.0]),
        glm_wt=jnp.array([1.0, 1.0]),
        converged=jnp.array(True),
        num_iters=jnp.array(1),
        objective=jnp.array(0.0),
        objective_delta=jnp.array(-1e-4),
        score_residual=jnp.array([0.0, 0.0]),
    )


def test_fit_passes_grammar_nouns_to_custom_fitter() -> None:
    seen: dict[str, object] = {}
    expected = _make_fit_result()
    current_fitted_glm_type = importlib.import_module("glmax._fit").FittedGLM

    class RecordingFitter(AbstractFitter, strict=True):
        solver: AbstractLinearSolver = CholeskySolver()

        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            seen["model"] = model
            seen["data"] = data
            seen["init"] = init
            return expected

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))
    init = Params(beta=jnp.zeros(1), disp=jnp.array(0.0))

    result = glmax.fit(model, data, init=init, fitter=RecordingFitter())

    assert isinstance(result, current_fitted_glm_type)
    assert result.result is expected
    assert seen["model"] is model
    assert seen["data"] is data
    assert seen["init"] is init


def test_fit_rejects_non_fitresult_from_custom_fitter() -> None:
    class BadFitter(AbstractFitter, strict=True):
        solver: AbstractLinearSolver = CholeskySolver()

        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> object:
            del model, data, init
            return object()

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))

    with pytest.raises(TypeError, match="FitResult"):
        glmax.fit(model, data, fitter=BadFitter())


_DEFAULT_X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
# Gamma uses InverseLink (eta = 1/mu); X must not include zero to avoid eta=0 at init.
_GAMMA_X = jnp.array([[1.0], [2.0], [3.0], [4.0]])


@pytest.mark.parametrize(
    ("family", "X", "y"),
    [
        (Gaussian(), _DEFAULT_X, jnp.array([0.2, 1.1, 2.0, 3.3])),
        (Poisson(), _DEFAULT_X, jnp.array([0.0, 1.0, 1.0, 2.0])),
        (Binomial(), _DEFAULT_X, jnp.array([0.0, 1.0, 0.0, 1.0])),
        (NegativeBinomial(), _DEFAULT_X, jnp.array([0.0, 1.0, 2.0, 4.0])),
        (Gamma(), _GAMMA_X, jr.gamma(jr.PRNGKey(1), 2.0, shape=(4,))),
    ],
)
def test_default_fitter_returns_canonical_fitresult_for_supported_families(family, X, y) -> None:
    current_fitted_glm_type = importlib.import_module("glmax._fit").FittedGLM
    model = glmax.specify(family=family)
    data = GLMData(X=X, y=y)

    result = glmax.fit(model, data)

    assert isinstance(result, current_fitted_glm_type)
    assert result.params.beta.shape == (1,)
    assert result.eta.shape == y.shape
    assert result.mu.shape == y.shape
    assert result.glm_wt.shape == y.shape
    assert result.score_residual.shape == y.shape


# ---------------------------------------------------------------------------
# [HIGH] JIT compatibility: FisherInfoError, wald_test
# Note: IRLSFitter itself is NOT JIT-safe (Python branching on family type).
# These tests cover the underlying kernels only.
# ---------------------------------------------------------------------------


def _make_gaussian_xy(n: int = 30, p: int = 2, seed: int = 0):
    key = jr.PRNGKey(seed)
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    y = X @ jnp.ones(p) + jr.normal(jr.PRNGKey(seed + 1), (n,)) * 0.2
    return X, y


def test_fisher_info_error_jit_is_finite():
    """FisherInfoError is JIT-safe: filter_jit output is finite."""
    from glmax._infer.stderr import FisherInfoError

    X, y = _make_gaussian_xy()
    family = Gaussian()
    model = glmax.specify(family=family)
    result = glmax.fit(model, GLMData(X=X, y=y))

    estimator = FisherInfoError()

    def _jit_fisher(result):
        return estimator(result)

    cov_jit = eqx.filter_jit(_jit_fisher)(result)

    assert bool(
        jnp.all(jnp.isfinite(cov_jit))
    ), f"JIT FisherInfoError covariance must be finite; got diag={jnp.diag(cov_jit)}"


def test_huber_error_is_finite_and_symmetric() -> None:
    from glmax._infer.stderr import HuberError

    X, y = _make_gaussian_xy()
    model = glmax.specify(family=Gaussian())
    result = glmax.fit(model, GLMData(X=X, y=y))

    cov = HuberError()(result)

    assert bool(jnp.all(jnp.isfinite(cov))), f"HuberError covariance must be finite; got diag={jnp.diag(cov)}"
    assert bool(jnp.allclose(cov, cov.T, atol=1e-6)), "HuberError covariance must be symmetric"


@pytest.mark.parametrize("family", [Gaussian(), Poisson()])
def test_wald_test_jit_is_finite(family):
    """wald_test is JIT-safe for Gaussian and Poisson: filter_jit output is finite."""
    from glmax._infer.hyptest import _wald_test

    stat = jnp.array([2.0, -1.5, 0.5])

    def _jit_wald(stat):
        return _wald_test(stat, 50, family)

    p_jit = eqx.filter_jit(_jit_wald)(stat)

    assert bool(
        jnp.all(jnp.isfinite(p_jit))
    ), f"JIT wald_test p-values must be finite for {type(family).__name__}; got {p_jit}"


def test_gaussian_se_equals_ols_formula() -> None:
    import jax.numpy.linalg as jla

    from glmax import GLMData
    from glmax.family import Gaussian

    key = jr.PRNGKey(7)
    n, p = 100, 3
    X = jnp.concatenate([jnp.ones((n, 1)), jr.normal(key, (n, p - 1))], axis=1)
    true_beta = jnp.array([2.0, 1.0, -0.5])
    sigma = 0.5
    y = X @ true_beta + jr.normal(jr.PRNGKey(8), (n,)) * sigma

    model = glmax.specify(family=Gaussian())
    result = glmax.fit(model, GLMData(X=X, y=y))
    inference = glmax.infer(result)

    family = Gaussian()
    phi = family.scale(result.X, result.y, result.mu)
    w_pure = result.glm_wt * phi
    information = (result.X * w_pure[:, jnp.newaxis]).T @ result.X
    expected_cov = phi * jla.inv(information)
    expected_se = jnp.sqrt(jnp.diag(expected_cov))

    assert jnp.allclose(
        inference.se, expected_se, atol=1e-5
    ), f"SE mismatch: got {inference.se}, expected {expected_se}"
