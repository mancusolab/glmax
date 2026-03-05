import importlib
import inspect

from pathlib import Path

import pytest

import jax.numpy as jnp

import glmax


def _sample_inputs():
    X = jnp.array([[1.0, 0.5], [1.0, -0.5], [1.0, 1.5], [1.0, -1.5]])
    y = jnp.array([1.0, 0.0, 2.0, 1.0])
    return X, y


def test_canonical_fit_rejects_invalid_fitter_type():
    X, y = _sample_inputs()

    with pytest.raises(TypeError, match="fitter must implement AbstractGLMFitter"):
        glmax.fit(X, y, family=glmax.Poisson(), solver=glmax.CholeskySolver(), fitter=object())


def test_glm_fit_rejects_invalid_override_fitter_type():
    X, y = _sample_inputs()
    model = glmax.GLM(family=glmax.Poisson(), solver=glmax.CholeskySolver())

    with pytest.raises(TypeError, match="fitter must implement AbstractGLMFitter"):
        model.fit(X, y, fitter="not-a-fitter")


def test_fitter_and_inference_docstrings_follow_contract_sections():
    fitters = importlib.import_module("glmax.infer.fitters")
    inference = importlib.import_module("glmax.infer.inference")

    targets = (
        fitters.AbstractGLMFitter,
        fitters.IRLSFitter,
        fitters.IRLSFitter.__call__,
        fitters.irls,
        inference.wald_test,
    )

    for target in targets:
        doc = inspect.getdoc(target)
        assert doc is not None
        assert "**Arguments:**" in doc
        assert "**Returns:**" in doc
        assert "**Raises:**" in doc or "**Failure Modes:**" in doc


def test_docs_claim_invalid_fitter_type_is_enforced():
    doc = Path("docs/api/glm.md").read_text()
    assert "invalid fitter type" in doc

    X, y = _sample_inputs()
    with pytest.raises(TypeError, match="fitter must implement AbstractGLMFitter"):
        glmax.fit(X, y, family=glmax.Poisson(), solver=glmax.CholeskySolver(), fitter=object())
