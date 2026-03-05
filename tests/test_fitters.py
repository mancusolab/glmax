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
