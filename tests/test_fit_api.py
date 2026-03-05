import inspect

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, FitResult, Fitter, GLMData, InferenceResult, Params
from glmax.fit import fit
from glmax.glm import specify


def test_canonical_contract_imports_exist() -> None:
    assert GLMData is not None
    assert Params is not None
    assert FitResult is not None
    assert InferenceResult is not None
    assert Diagnostics is not None
    assert Fitter is not None


def _make_fit_result() -> FitResult:
    return FitResult(
        params=Params(beta=jnp.array([1.0]), disp=jnp.array(0.0)),
        se=jnp.array([0.5]),
        z=jnp.array([2.0]),
        p=jnp.array([0.05]),
        eta=jnp.array([1.0]),
        mu=jnp.array([1.0]),
        glm_wt=jnp.array([1.0]),
        diagnostics=Diagnostics(converged=jnp.array(True), num_iters=jnp.array(1)),
        infor_inv=jnp.array([[1.0]]),
        resid=jnp.array([0.0]),
    )


def test_fit_signature_matches_canonical_surface() -> None:
    sig = inspect.signature(fit)
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

    result = fit(model, data, init=init, fitter=DummyFitter())

    assert result is expected
    assert isinstance(result, FitResult)
    assert seen["model"] is model
    assert seen["data"] is data
    assert seen["init"] is init


def test_specify_returns_glm_instance() -> None:
    model = specify()
    assert isinstance(model, glmax.GLM)
