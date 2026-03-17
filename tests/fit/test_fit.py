# pattern: Imperative Shell

import importlib

import pytest

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import glmax

from glmax import AbstractFitter, FitResult, FittedGLM, GLMData, Params
from glmax._fit import IRLSFitter
from glmax._fit.solve import AbstractLinearSolver, CholeskySolver, QRSolver
from glmax.family import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson
from glmax.glm import specify


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
    assert bool(eqx.tree_equal(result.result, expected))
    assert isinstance(seen["model"], glmax.GLM)
    assert isinstance(seen["data"], GLMData)
    assert isinstance(seen["init"], Params)


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


def test_fit_boundary_rejects_raw_data_and_non_params_init() -> None:
    with pytest.raises(TypeError, match="GLM"):
        glmax.fit(object(), GLMData(X=jnp.ones((3, 1)), y=jnp.ones(3)))

    with pytest.raises(TypeError, match="GLMData"):
        glmax.fit(glmax.GLM(), jnp.ones((3, 1)))

    with pytest.raises(TypeError, match="Params"):
        glmax.fit(glmax.GLM(), GLMData(X=jnp.ones((3, 1)), y=jnp.ones(3)), init=jnp.zeros(1))


def test_fitter_is_abstract_equinox_model() -> None:
    assert issubclass(AbstractFitter, eqx.Module)

    with pytest.raises(TypeError):
        AbstractFitter()


def test_canonical_fit_supports_non_default_solver_path() -> None:
    model = glmax.specify(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.0], [1.0, 3.0]]),
        y=jnp.array([0.8, 1.7, 2.1, 2.9]),
    )

    result = glmax.fit(model, data, fitter=IRLSFitter(solver=QRSolver()))

    assert isinstance(result, FittedGLM)
    assert result.params.beta.shape == (2,)
    assert bool(result.converged)
    assert jnp.all(jnp.isfinite(result.params.beta))


def test_default_fitter_forwards_offset_and_transforms_init_to_eta() -> None:
    model = glmax.GLM(family=Gaussian())
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]])
    y = jnp.array([1.0, 0.0, 1.0])
    offset = jnp.array([0.2, 0.1, 0.3])
    init = Params(beta=jnp.array([0.4, -0.1]), disp=jnp.array(0.7))

    data = GLMData(X=X, y=y, offset=offset)
    result_1 = glmax.fit(model, data, init=init)
    result_2 = glmax.fit(model, data, init=init)

    assert isinstance(result_1, FittedGLM)
    assert jnp.allclose(result_1.beta, result_2.beta)
    assert jnp.allclose(result_1.params.disp, result_2.params.disp)


def test_default_fitter_validates_init_beta_shape() -> None:
    X = jnp.ones((4, 2))
    y = jnp.ones(4)
    bad_init = Params(beta=jnp.ones((2, 1)), disp=jnp.array(0.0))

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.fit(glmax.GLM(), GLMData(X=X, y=y), init=bad_init)


@pytest.mark.parametrize(
    ("family", "y"),
    [
        (Gaussian(), jnp.array([0.1, 1.0, 2.1, 2.9, 4.2])),
        (Poisson(), jnp.array([0.0, 1.0, 1.0, 2.0, 3.0])),
        (Binomial(), jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])),
        (NegativeBinomial(), jnp.array([0.0, 1.0, 2.0, 1.0, 4.0])),
    ],
)
def test_all_families_succeed_with_default_fitter(family, y) -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), y=y)
    result = glmax.fit(glmax.GLM(family=family), data)

    assert isinstance(result, FittedGLM)
    assert isinstance(result.params, Params)
    assert result.score_residual.shape == (data.n_samples,)
    assert bool(jnp.isfinite(result.objective))
    assert bool(jnp.isfinite(result.objective_delta))
    if isinstance(family, (NegativeBinomial, Gaussian)):
        assert result.params.disp > 0
    else:
        assert jnp.allclose(result.params.disp, jnp.array(1.0))


def test_single_feature_beta_shape_roundtrip() -> None:
    model = glmax.GLM(family=Gaussian())
    X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    y = jnp.array([1.2, 1.9, 3.1, 4.0])
    data = GLMData(X=X, y=y)

    first = glmax.fit(model, data)
    assert first.beta.shape == (1,)

    second = glmax.fit(model, data, init=first.params)
    assert second.beta.shape == (1,)


def test_unsupported_weights_rejected() -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    y = jnp.array([0.2, 0.9, 2.2, 2.8])

    with pytest.raises(ValueError, match="weights"):
        glmax.fit(glmax.GLM(), GLMData(X=X, y=y, weights=jnp.ones(4)))


def test_specify_returns_glm_instance() -> None:
    model = specify()
    assert isinstance(model, glmax.GLM)
