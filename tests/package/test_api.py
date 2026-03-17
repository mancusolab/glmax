# pattern: Imperative Shell

import inspect

from pathlib import Path

import pytest

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

import glmax

from glmax import AbstractFitter, Diagnostics, FitResult, FittedGLM, GLMData, InferenceResult, Params
from glmax._fit import IRLSFitter
from glmax._fit.solve import AbstractLinearSolver, CholeskySolver, QRSolver
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_INIT = WORKTREE_ROOT / "src" / "glmax" / "__init__.py"


def _make_fit_result() -> FitResult:
    return FitResult(
        params=Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux=jnp.array(0.25)),
        X=jnp.array([[1.0]]),
        y=jnp.array([1.0]),
        eta=jnp.array([1.0]),
        mu=jnp.array([1.0]),
        glm_wt=jnp.array([1.0]),
        converged=jnp.array(True),
        num_iters=jnp.array(1),
        objective=jnp.array(0.1),
        objective_delta=jnp.array(-1e-3),
        score_residual=jnp.array([0.0]),
    )


def test_canonical_contract_imports_exist() -> None:
    assert GLMData is not None
    assert Params is not None
    assert FitResult is not None
    assert FittedGLM is not None
    assert InferenceResult is not None
    assert Diagnostics is not None
    assert AbstractFitter is not None


def test_fitter_is_abstract_equinox_model() -> None:
    assert issubclass(AbstractFitter, eqx.Module)

    with pytest.raises(TypeError):
        AbstractFitter()


def test_top_level_exports_are_canonical_nouns_and_verbs() -> None:
    assert set(glmax.__all__) == {
        "GLMData",
        "Params",
        "GLM",
        "AbstractFitter",
        "FitResult",
        "FittedGLM",
        "InferenceResult",
        "Diagnostics",
        "AbstractTest",
        "WaldTest",
        "ScoreTest",
        "AbstractStdErrEstimator",
        "FisherInfoError",
        "HuberError",
        "specify",
        "predict",
        "fit",
        "infer",
        "check",
    }


def test_top_level_fit_resolves_to_canonical_entrypoint() -> None:
    assert callable(glmax.fit)
    assert glmax.fit.__module__ == "glmax._fit.fit"


def test_pytest_imports_glmax_from_worktree_src() -> None:
    assert Path(glmax.__file__).resolve() == EXPECTED_INIT.resolve()


def test_fit_signature_matches_canonical_surface() -> None:
    sig = inspect.signature(glmax.fit)
    assert list(sig.parameters) == ["model", "data", "init", "fitter"]
    assert sig.parameters["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["init"].default is None
    assert sig.parameters["fitter"].kind is inspect.Parameter.KEYWORD_ONLY


def test_infer_signature_matches_canonical_surface() -> None:
    sig = inspect.signature(glmax.infer)
    assert list(sig.parameters) == ["fitted", "inferrer", "stderr"]
    assert sig.parameters["fitted"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["inferrer"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["stderr"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["inferrer"].default is not None


def test_fit_returns_fittedglm_using_injected_fitter() -> None:
    expected = _make_fit_result()
    seen: dict[str, object] = {}

    class DummyFitter(AbstractFitter, strict=True):
        solver: AbstractLinearSolver = CholeskySolver()

        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            seen["model"] = model
            seen["data"] = data
            seen["init"] = init
            return expected

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))
    init = Params(beta=jnp.zeros(1), disp=jnp.array(0.0), aux=None)

    result = glmax.fit(model, data, init=init, fitter=DummyFitter())

    assert isinstance(result, FittedGLM)
    assert bool(eqx.tree_equal(result.model, model))
    assert bool(eqx.tree_equal(result.result, expected))
    assert isinstance(seen["model"], glmax.GLM)
    assert isinstance(seen["data"], GLMData)
    assert isinstance(seen["init"], Params)
    assert seen["init"]._fields == ("beta", "disp", "aux")


def test_fit_rejects_non_fitter_with_deterministic_error() -> None:
    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))

    with pytest.raises(TypeError, match=r"expects `fitter` to be an AbstractFitter instance"):
        glmax.fit(model, data, fitter="not-a-fitter")


def test_contract_dataclasses_are_pytrees() -> None:
    params = Params(beta=jnp.array([1.0, 2.0]), disp=jnp.array(0.5), aux=jnp.array(0.25))
    leaves, tree = jtu.tree_flatten(params)
    assert len(leaves) == 3
    rebuilt = jtu.tree_unflatten(tree, leaves)
    assert jnp.allclose(rebuilt.beta, params.beta)
    assert jnp.allclose(rebuilt.disp, params.disp)
    assert jnp.allclose(rebuilt.aux, params.aux)

    result = _make_fit_result()
    fit_leaves, _ = jtu.tree_flatten(result)
    assert len(fit_leaves) == 13
    assert not hasattr(result, "information")
    assert not hasattr(result, "infor_inv")
    assert not hasattr(result, "resid")
    assert not hasattr(result, "se")
    assert not hasattr(result, "z")
    assert not hasattr(result, "p")


def test_canonical_fit_supports_non_default_solver_constructor_path() -> None:
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


@pytest.mark.parametrize(
    ("family", "y"),
    [
        (Gaussian(), jnp.array([0.1, 1.0, 2.1, 2.9, 4.2])),
        (Poisson(), jnp.array([0.0, 1.0, 1.0, 2.0, 3.0])),
        (Binomial(), jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])),
        (NegativeBinomial(), jnp.array([0.0, 1.0, 2.0, 1.0, 4.0])),
    ],
)
def test_canonical_fit_succeeds_for_supported_families(family, y) -> None:
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


def test_predict_rejects_invalid_params_contracts_deterministically() -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.predict(model, Params(beta=jnp.array([1.0, 2.0]), disp=jnp.array(0.0), aux=None), data)

    with pytest.raises(TypeError, match="Params.beta must be numeric"):
        glmax.predict(model, Params(beta=["bad"], disp=jnp.array(0.0), aux=None), data)

    with pytest.raises(TypeError, match="Params.disp must be numeric"):
        glmax.predict(model, Params(beta=jnp.array([1.0]), disp="bad", aux=None), data)

    with pytest.raises(TypeError, match="Params.beta must have an inexact dtype"):
        glmax.predict(model, Params(beta=jnp.array([1], dtype=jnp.int32), disp=jnp.array(0.0), aux=None), data)

    with pytest.raises(TypeError, match="Params.disp must have an inexact dtype"):
        glmax.predict(model, Params(beta=jnp.array([1.0]), disp=jnp.array(0, dtype=jnp.int32), aux=None), data)

    with pytest.raises(TypeError, match="Params.aux must be numeric"):
        glmax.predict(model, Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux="bad"), data)

    with pytest.raises(TypeError, match="Params.aux must have an inexact dtype"):
        glmax.predict(
            model,
            Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux=jnp.array(0, dtype=jnp.int32)),
            data,
        )

    with pytest.raises(ValueError, match="Params.aux must be a scalar"):
        glmax.predict(model, Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux=jnp.array([0.1, 0.2])), data)


def test_predict_rejects_aux_for_families_without_aux_state() -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))
    params = Params(beta=jnp.array([1.0]), disp=jnp.array(1.0), aux=jnp.array(0.25))

    with pytest.raises(ValueError, match="Gaussian does not support auxiliary parameters"):
        glmax.predict(model, params, data)


def test_single_feature_fit_keeps_beta_vector_shape_for_roundtrip_init() -> None:
    from typing import ClassVar

    from glmax.family.dist import ExponentialDispersionFamily
    from glmax.family.links import IdentityLink

    class _CanonicalWarmStartFamily(ExponentialDispersionFamily):
        glink: IdentityLink = IdentityLink()
        _links: ClassVar[list[type[IdentityLink]]] = [IdentityLink]
        _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

        def scale(self, X, y, mu):
            del X, y, mu
            return jnp.asarray(1.0)

        def negloglikelihood(self, y, eta, disp=1.0):
            resid = jnp.asarray(y) - jnp.asarray(eta)
            safe_disp = self.canonical_dispersion(disp)
            return jnp.sum(jnp.square(resid)) / safe_disp + safe_disp

        def variance(self, mu, disp=1.0):
            safe_disp = self.canonical_dispersion(disp)
            return jnp.ones_like(jnp.asarray(mu)) * safe_disp

        def sample(self, key, eta, disp=1.0):
            del key, disp
            return jnp.asarray(eta)

        def update_dispersion(self, X, y, eta, disp=1.0, step_size=1.0):
            del X, y, eta, step_size
            return jnp.asarray(disp)

        def estimate_dispersion(self, X, y, eta, disp=1.0, step_size=1.0, tol=1e-3, max_iter=1000, offset_eta=0.0):
            del X, y, eta, step_size, tol, max_iter, offset_eta
            return jnp.asarray(disp)

        def canonical_dispersion(self, disp=0.0):
            return jnp.maximum(jnp.asarray(disp), jnp.asarray(1.0))

        def canonical_auxiliary(self, aux=None):
            if aux is None:
                return jnp.asarray(0.25)
            return jnp.maximum(jnp.asarray(aux), jnp.asarray(0.25))

    model = glmax.GLM(family=_CanonicalWarmStartFamily())
    X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    y = jnp.array([1.2, 1.9, 3.1, 4.0])
    data = GLMData(X=X, y=y)
    seed = Params(beta=jnp.zeros(1), disp=jnp.array(0.7), aux=jnp.array(0.1))

    first = glmax.fit(model, data, init=seed)
    assert first.beta.shape == (1,)
    assert jnp.allclose(first.params.disp, jnp.array(1.0))
    assert jnp.allclose(first.params.aux, jnp.array(0.25))

    inferred = glmax.infer(first)
    second = glmax.fit(model, data, init=first.params)
    assert second.beta.shape == (1,)
    assert jnp.allclose(inferred.params.disp, first.params.disp)
    assert jnp.allclose(inferred.params.aux, first.params.aux)
    assert jnp.allclose(second.params.disp, first.params.disp)
    assert jnp.allclose(second.params.aux, first.params.aux)
