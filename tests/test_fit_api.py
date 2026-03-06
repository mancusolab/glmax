# pattern: Imperative Shell
import importlib
import inspect
import os
import subprocess
import sys

from pathlib import Path

import pytest

import jax.numpy as jnp
import jax.tree_util as jtu

import glmax

from glmax import Diagnostics, FitResult, Fitter, GLMData, InferenceResult, Params
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson
from glmax.glm import specify
from glmax.infer.solve import QRSolver


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INIT = WORKTREE_ROOT / "src" / "glmax" / "__init__.py"


def test_canonical_contract_imports_exist() -> None:
    assert GLMData is not None
    assert Params is not None
    assert FitResult is not None
    assert InferenceResult is not None
    assert Diagnostics is not None
    assert Fitter is not None


def test_top_level_exports_are_canonical_nouns_and_verbs() -> None:
    assert set(glmax.__all__) == {
        "GLMData",
        "Params",
        "GLM",
        "Fitter",
        "FitResult",
        "InferenceResult",
        "Diagnostics",
        "specify",
        "predict",
        "fit",
        "infer",
        "check",
    }


def test_top_level_fit_resolves_to_canonical_entrypoint() -> None:
    assert callable(glmax.fit)
    assert glmax.fit.__module__ == "glmax.fit"


def test_pytest_imports_glmax_from_worktree_src() -> None:
    assert Path(glmax.__file__).resolve() == EXPECTED_INIT.resolve()


def test_worktree_python_wrapper_imports_glmax_from_worktree_src() -> None:
    command = [
        str(WORKTREE_ROOT / "tools" / "worktree-python"),
        "-c",
        "import glmax, pathlib; print(pathlib.Path(glmax.__file__).resolve())",
    ]
    env = os.environ.copy()
    env["GLMAX_PYTHON_BIN"] = sys.executable
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=WORKTREE_ROOT,
        env=env,
    )

    assert completed.stdout.strip() == str(EXPECTED_INIT.resolve())


def test_legacy_fit_state_alias_is_not_publicly_exported() -> None:
    assert not hasattr(glmax, "GLMState")
    assert "GLMState" not in glmax.__all__


def test_infer_shims_are_not_publicly_reexported() -> None:
    infer_module = importlib.import_module("glmax.infer")

    assert not hasattr(infer_module, "irls")
    assert not hasattr(infer_module, "AbstractFitter")
    assert not hasattr(infer_module, "DefaultFitter")
    assert not hasattr(infer_module, "CholeskySolver")
    assert not hasattr(infer_module, "QRSolver")
    assert not hasattr(infer_module, "CGSolver")
    assert not hasattr(infer_module, "FisherInfoError")
    assert not hasattr(infer_module, "HuberError")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax.infer.state")


def _make_fit_result() -> FitResult:
    return FitResult(
        params=Params(beta=jnp.array([1.0]), disp=jnp.array(0.0)),
        se=jnp.array([0.5]),
        z=jnp.array([2.0]),
        p=jnp.array([0.05]),
        eta=jnp.array([1.0]),
        mu=jnp.array([1.0]),
        glm_wt=jnp.array([1.0]),
        diagnostics=Diagnostics(
            converged=jnp.array(True),
            num_iters=jnp.array(1),
            objective=jnp.array(0.1),
            objective_delta=jnp.array(-1e-3),
        ),
        curvature=jnp.array([[1.0]]),
        score_residual=jnp.array([0.0]),
    )


def test_fit_signature_matches_canonical_surface() -> None:
    sig = inspect.signature(glmax.fit)
    assert list(sig.parameters) == ["model", "data", "init", "fitter"]
    assert sig.parameters["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["data"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["init"].default is None
    assert sig.parameters["fitter"].kind is inspect.Parameter.KEYWORD_ONLY


def test_fit_returns_fitresult_using_injected_fitter() -> None:
    expected = _make_fit_result()
    seen: dict[str, object] = {}

    class DummyFitter:
        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            seen["model"] = model
            seen["data"] = data
            seen["init"] = init
            return expected

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))
    init = Params(beta=jnp.zeros(1), disp=jnp.array(0.0))

    result = glmax.fit(model, data, init=init, fitter=DummyFitter())

    assert result is expected
    assert isinstance(result, FitResult)
    assert seen["model"] is model
    assert seen["data"] is data
    assert seen["init"] is init


def test_fit_rejects_non_callable_fitter_with_deterministic_error() -> None:
    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))

    with pytest.raises(TypeError, match=r"expects `fitter` to be callable"):
        glmax.fit(model, data, fitter="not-a-fitter")


def test_default_fitter_forwards_offset_and_transforms_init_to_eta() -> None:
    model = glmax.GLM(family=Gaussian())
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]])
    y = jnp.array([1.0, 0.0, 1.0])
    offset = jnp.array([0.2, 0.1, 0.3])
    init = Params(beta=jnp.array([0.4, -0.1]), disp=jnp.array(0.7))

    data = GLMData(X=X, y=y, offset=offset)
    result_1 = glmax.fit(model, data, init=init)
    result_2 = glmax.fit(model, data, init=init)

    assert isinstance(result_1, FitResult)
    assert jnp.allclose(result_1.beta, result_2.beta)
    assert jnp.allclose(result_1.params.disp, result_2.params.disp)


def test_default_top_level_fit_does_not_dispatch_through_model_fit_override() -> None:
    class OverrideRaisesGLM(glmax.GLM):
        def fit(self, data, **kwargs):
            del data, kwargs
            raise AssertionError("top-level fit should not dispatch through GLM.fit overrides")

    model = OverrideRaisesGLM(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]]),
        y=jnp.array([1.0, 0.0, 1.0]),
        offset=jnp.array([0.2, 0.1, 0.3]),
    )
    init = Params(beta=jnp.array([0.4, -0.1]), disp=jnp.array(0.7))

    result = glmax.fit(model, data, init=init)

    assert isinstance(result, FitResult)
    assert result.params.beta.shape == (2,)
    assert jnp.ndim(result.params.disp) == 0


def test_contract_dataclasses_are_pytrees() -> None:
    params = Params(beta=jnp.array([1.0, 2.0]), disp=jnp.array(0.5))
    leaves, tree = jtu.tree_flatten(params)
    assert len(leaves) == 2
    rebuilt = jtu.tree_unflatten(tree, leaves)
    assert jnp.allclose(rebuilt.beta, params.beta)
    assert jnp.allclose(rebuilt.disp, params.disp)

    result = _make_fit_result()
    fit_leaves, _ = jtu.tree_flatten(result)
    assert len(fit_leaves) == 14
    assert not hasattr(result, "infor_inv")
    assert not hasattr(result, "resid")


def test_default_fitter_rejects_unsupported_weights_and_mask() -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    y = jnp.array([0.2, 0.9, 2.2, 2.8])

    with pytest.raises(ValueError, match="weights"):
        glmax.fit(glmax.GLM(), GLMData(X=X, y=y, weights=jnp.ones(4)))

    masked_result = glmax.fit(
        glmax.GLM(family=Gaussian()),
        GLMData(X=X, y=y, mask=jnp.array([True, False, True, True])),
    )
    assert isinstance(masked_result, FitResult)


def test_fit_boundary_rejects_raw_data_and_non_params_init() -> None:
    with pytest.raises(TypeError, match="GLM"):
        glmax.fit(object(), GLMData(X=jnp.ones((3, 1)), y=jnp.ones(3)))

    with pytest.raises(TypeError, match="GLMData"):
        glmax.fit(glmax.GLM(), jnp.ones((3, 1)))

    with pytest.raises(TypeError, match="Params"):
        glmax.fit(glmax.GLM(), GLMData(X=jnp.ones((3, 1)), y=jnp.ones(3)), init=jnp.zeros(1))


def test_glm_fit_is_not_a_curated_public_contract() -> None:
    doc = glmax.GLM.fit.__doc__ or ""

    assert not hasattr(glmax.GLM.fit, "__signature__")
    assert "Use `glmax.fit(model, data, init=...)` for the public grammar contract." in doc


def test_glm_fit_signature_does_not_expose_legacy_wrapper_parameters() -> None:
    sig = inspect.signature(glmax.GLM.fit)

    assert "legacy_args" not in sig.parameters
    assert "legacy_kwargs" not in sig.parameters
    assert "init" not in sig.parameters
    assert "alpha_init" not in sig.parameters
    assert all(param.kind is not inspect.Parameter.VAR_POSITIONAL for param in sig.parameters.values())
    assert all(param.kind is not inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())


def test_bound_glm_fit_signature_does_not_expose_model_parameter() -> None:
    sig = inspect.signature(glmax.GLM(family=Gaussian()).fit)

    assert list(sig.parameters) == [
        "data",
        "init_eta",
        "disp_init",
        "se_estimator",
        "max_iter",
        "tol",
        "step_size",
    ]
    assert "model" not in sig.parameters


@pytest.mark.parametrize(
    ("legacy_keyword", "match"),
    [
        ("init", r"GLM\.fit\(\.\.\.\) no longer accepts `init`.*glmax\.fit\(model, data, init=\.\.\.\)"),
        ("alpha_init", r"GLM\.fit\(\.\.\.\) no longer accepts `alpha_init`.*Use `disp_init=` instead"),
    ],
)
def test_glm_fit_removed_legacy_keywords_raise_migration_typeerrors(legacy_keyword: str, match: str) -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.0, 1.0, 2.0, 3.0]))

    with pytest.raises(TypeError, match=match):
        model.fit(data, **{legacy_keyword: jnp.zeros(1)})


@pytest.mark.parametrize(
    ("extra_arg", "match"),
    [
        (
            jnp.zeros(4),
            r"GLM\.fit\(\.\.\.\) accepts exactly one positional argument after binding: `data`.*Use `init_eta=`",
        ),
        (
            Params(beta=jnp.zeros(1), disp=jnp.array(0.0)),
            r"GLM\.fit\(\.\.\.\) no longer accepts positional `Params`.*Use `glmax\.fit\(model, data, init=params\)`",
        ),
    ],
)
def test_glm_fit_rejects_legacy_extra_positional_arguments_with_migration_guidance(
    extra_arg: object, match: str
) -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.0, 1.0, 2.0, 3.0]))

    with pytest.raises(TypeError, match=match):
        model.fit(data, extra_arg)


def test_canonical_fit_supports_non_default_solver_constructor_path() -> None:
    model = glmax.specify(family=Gaussian(), solver=QRSolver())
    data = GLMData(
        X=jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.0], [1.0, 3.0]]),
        y=jnp.array([0.8, 1.7, 2.1, 2.9]),
    )

    result = glmax.fit(model, data)

    assert isinstance(result, FitResult)
    assert result.params.beta.shape == (2,)
    assert bool(result.converged)
    assert jnp.all(jnp.isfinite(result.params.beta))


def test_legacy_array_first_fitter_module_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax.infer.fitter")


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

    assert isinstance(result, FitResult)
    assert isinstance(result.params, Params)
    assert result.curvature.shape == (1, 1)
    assert result.score_residual.shape == (data.n_samples,)
    assert bool(jnp.isfinite(result.objective))
    assert bool(jnp.isfinite(result.objective_delta))
    if isinstance(family, NegativeBinomial):
        assert result.params.disp > 0
    else:
        assert jnp.allclose(result.params.disp, jnp.array(0.0))


def test_fit_boundary_rejects_non_finite_params_init() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="finite"):
        glmax.fit(glmax.GLM(family=Gaussian()), data, init=Params(beta=jnp.array([jnp.nan]), disp=jnp.array(0.0)))

    with pytest.raises(ValueError, match="finite"):
        glmax.fit(glmax.GLM(family=Gaussian()), data, init=Params(beta=jnp.array([0.0]), disp=jnp.array(jnp.inf)))


def test_default_fitter_validates_init_beta_shape() -> None:
    X = jnp.ones((4, 2))
    y = jnp.ones(4)
    bad_init = Params(beta=jnp.ones((2, 1)), disp=jnp.array(0.0))

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.fit(glmax.GLM(), GLMData(X=X, y=y), init=bad_init)


def test_default_fitter_validates_scalar_disp() -> None:
    X = jnp.ones((4, 2))
    y = jnp.ones(4)
    bad_init = Params(beta=jnp.ones(2), disp=jnp.ones(2))

    with pytest.raises(ValueError, match="Params.disp"):
        glmax.fit(glmax.GLM(), GLMData(X=X, y=y), init=bad_init)


def test_fit_rejects_malformed_fitresult_from_custom_fitter() -> None:
    class BadFitter:
        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            del model, data, init
            return FitResult(
                params=Params(beta=jnp.array([1.0]), disp=jnp.array(0.0)),
                se=jnp.array([0.5]),
                z=jnp.array([2.0]),
                p=jnp.array([0.05]),
                eta=jnp.array([1.0]),
                mu=jnp.array([1.0]),
                glm_wt=jnp.array([1.0]),
                diagnostics=Diagnostics(
                    converged=jnp.array(True),
                    num_iters=jnp.array(1),
                    objective=jnp.array(0.0),
                    objective_delta=jnp.array(-1e-3),
                ),
                curvature=jnp.array([[jnp.nan]]),
                score_residual=jnp.array([0.0]),
            )

    with pytest.raises(ValueError, match="FitResult.curvature"):
        glmax.fit(
            glmax.GLM(),
            GLMData(X=jnp.array([[1.0]]), y=jnp.array([1.0])),
            fitter=BadFitter(),
        )


def test_predict_rejects_invalid_params_contracts_deterministically() -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.predict(model, Params(beta=jnp.array([1.0, 2.0]), disp=jnp.array(0.0)), data)

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.predict(model, Params(beta=jnp.array([jnp.nan]), disp=jnp.array(0.0)), data)

    with pytest.raises(TypeError, match="Params.beta must be numeric"):
        glmax.predict(model, Params(beta=["bad"], disp=jnp.array(0.0)), data)

    with pytest.raises(TypeError, match="Params.disp must be numeric"):
        glmax.predict(model, Params(beta=jnp.array([1.0]), disp="bad"), data)

    with pytest.raises(TypeError, match="Params.beta must have an inexact dtype"):
        glmax.predict(model, Params(beta=jnp.array([1], dtype=jnp.int32), disp=jnp.array(0.0)), data)

    with pytest.raises(TypeError, match="Params.disp must have an inexact dtype"):
        glmax.predict(model, Params(beta=jnp.array([1.0]), disp=jnp.array(0, dtype=jnp.int32)), data)


def test_default_fitter_validates_X_y_and_offset_shapes() -> None:
    with pytest.raises(ValueError, match="GLMData.X"):
        glmax.fit(glmax.GLM(), GLMData(X=jnp.ones(4), y=jnp.ones(4)))

    with pytest.raises(ValueError, match="GLMData.y"):
        glmax.fit(glmax.GLM(), GLMData(X=jnp.ones((4, 1)), y=jnp.ones((4, 1))))

    with pytest.raises(ValueError, match="GLMData.y"):
        glmax.fit(glmax.GLM(), GLMData(X=jnp.ones((4, 1)), y=jnp.ones(3)))

    with pytest.raises(ValueError, match="GLMData.offset"):
        glmax.fit(glmax.GLM(), GLMData(X=jnp.ones((4, 1)), y=jnp.ones(4), offset=jnp.ones((4, 1))))

    with pytest.raises(ValueError, match="GLMData.offset"):
        glmax.fit(glmax.GLM(), GLMData(X=jnp.ones((4, 1)), y=jnp.ones(4), offset=jnp.ones(3)))


def test_single_feature_fit_keeps_beta_vector_shape_for_roundtrip_init() -> None:
    model = glmax.GLM(family=Gaussian())
    X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    y = jnp.array([1.2, 1.9, 3.1, 4.0])
    data = GLMData(X=X, y=y)

    first = glmax.fit(model, data)
    assert first.beta.shape == (1,)

    second = glmax.fit(model, data, init=first.params)
    assert second.beta.shape == (1,)


def test_specify_returns_glm_instance() -> None:
    model = specify()
    assert isinstance(model, glmax.GLM)
