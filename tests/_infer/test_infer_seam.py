# pattern: Imperative Shell
import importlib
import sys

import pytest

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, FitResult, FittedGLM, GLMData, Params
from glmax._fit import IRLSFitter
from glmax._fit.solve import QRSolver
from glmax.family import Gaussian


STALE_INFER_MODULES = (
    "glmax._infer.fitter",
    "glmax._infer.fitters",
    "glmax._infer.result",
)


def _drop_stale_infer_modules() -> None:
    for name in STALE_INFER_MODULES:
        sys.modules.pop(name, None)


def _make_fit_result() -> FitResult:
    return FitResult(
        params=Params(beta=jnp.array([1.0]), disp=jnp.array(0.0)),
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


def test_glm_defaults_to_solver_from_fit_solve() -> None:
    solve_module = importlib.import_module("glmax._fit.solve")
    default_fitter = IRLSFitter()

    assert isinstance(default_fitter.solver, solve_module.AbstractLinearSolver)
    assert isinstance(default_fitter.solver, solve_module.CholeskySolver)


def test_qr_solver_import_path_remains_supported() -> None:
    solve_module = importlib.import_module("glmax._fit.solve")

    assert solve_module.QRSolver is QRSolver
    assert isinstance(QRSolver(), solve_module.AbstractLinearSolver)


def test_duplicate_solver_modules_are_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax._infer.contracts")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax._infer.solve")


def test_legacy_fitter_modules_are_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax._infer.fitter")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax._infer.fitters")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("glmax._infer.result")


def test_importing_canonical_fit_module_does_not_touch_stale_infer_modules() -> None:
    _drop_stale_infer_modules()
    fit_module = importlib.reload(importlib.import_module("glmax._fit"))

    assert not hasattr(fit_module, "_run_default_pipeline")
    assert all(name not in sys.modules for name in STALE_INFER_MODULES)


def test_canonical_fit_execution_does_not_import_stale_infer_modules() -> None:
    _drop_stale_infer_modules()
    current_fitted_glm_type = importlib.import_module("glmax._fit").FittedGLM

    result = glmax.fit(
        glmax.GLM(family=Gaussian()),
        GLMData(
            X=jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.0], [1.0, 3.0]]),
            y=jnp.array([0.8, 1.7, 2.1, 2.9]),
        ),
        fitter=IRLSFitter(solver=QRSolver()),
    )

    assert isinstance(result, current_fitted_glm_type)
    assert all(name not in sys.modules for name in STALE_INFER_MODULES)


def test_canonical_infer_and_check_do_not_import_stale_infer_modules() -> None:
    _drop_stale_infer_modules()
    fit_result = _make_fit_result()
    fitted = FittedGLM(model=glmax.GLM(family=Gaussian()), result=fit_result)

    inference_result = glmax.infer(fitted)
    diagnostics = glmax.check(fitted)

    assert isinstance(inference_result, glmax.InferenceResult)
    assert isinstance(diagnostics, Diagnostics)
    assert all(name not in sys.modules for name in STALE_INFER_MODULES)
