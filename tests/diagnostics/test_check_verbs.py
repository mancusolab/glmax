# pattern: Imperative Shell


import pytest

import jax.numpy as jnp

import glmax

from glmax import Diagnostics, GLMData
from glmax.family import Gaussian


def _make_fitted():
    model = glmax.specify(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        y=jnp.array([1.2, 1.9, 3.1, 4.2]),
    )
    return glmax.fit(model, data)


def test_check_returns_diagnostics_without_refitting() -> None:
    fitted = _make_fitted()

    diagnostics = glmax.check(fitted)

    assert isinstance(diagnostics, Diagnostics)
    assert diagnostics == Diagnostics()


def test_check_rejects_non_fitted_glm() -> None:
    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.check(object())
