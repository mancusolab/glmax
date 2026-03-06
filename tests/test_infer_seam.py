# pattern: Imperative Shell
import importlib
import sys

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, FitResult, GLMData, Params
from glmax.family import Gaussian
from glmax.infer.solve import QRSolver


STALE_INFER_MODULES = (
    "glmax.infer.fitter",
    "glmax.infer.fitters",
    "glmax.infer.result",
)


def _drop_stale_infer_modules() -> None:
    for name in STALE_INFER_MODULES:
        sys.modules.pop(name, None)


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


def test_glm_defaults_to_solver_from_infer_solve() -> None:
    model = glmax.GLM()
    solve_module = importlib.import_module("glmax.infer.solve")

    assert isinstance(model.solver, solve_module.AbstractLinearSolver)
    assert isinstance(model.solver, solve_module.CholeskySolver)


def test_qr_solver_import_path_remains_supported() -> None:
    solve_module = importlib.import_module("glmax.infer.solve")

    assert solve_module.QRSolver is QRSolver
    assert isinstance(QRSolver(), solve_module.AbstractLinearSolver)


def test_importing_canonical_fit_module_does_not_touch_stale_infer_modules() -> None:
    _drop_stale_infer_modules()
    fit_module = importlib.reload(importlib.import_module("glmax.fit"))

    assert not hasattr(fit_module, "_run_default_pipeline")
    assert all(name not in sys.modules for name in STALE_INFER_MODULES)


def test_canonical_fit_execution_does_not_import_stale_infer_modules() -> None:
    _drop_stale_infer_modules()

    result = glmax.fit(
        glmax.GLM(family=Gaussian(), solver=QRSolver()),
        GLMData(
            X=jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.0], [1.0, 3.0]]),
            y=jnp.array([0.8, 1.7, 2.1, 2.9]),
        ),
    )

    assert isinstance(result, FitResult)
    assert all(name not in sys.modules for name in STALE_INFER_MODULES)


def test_canonical_infer_and_check_do_not_import_stale_infer_modules() -> None:
    _drop_stale_infer_modules()
    model = glmax.GLM(family=Gaussian())
    fit_result = _make_fit_result()

    inference_result = glmax.infer(model, fit_result)
    diagnostics = glmax.check(model, fit_result)

    assert isinstance(inference_result, glmax.InferenceResult)
    assert isinstance(diagnostics, Diagnostics)
    assert all(name not in sys.modules for name in STALE_INFER_MODULES)
