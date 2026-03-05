import pytest

import jax.numpy as jnp

import glmax

from glmax import GLMData, Params
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


@pytest.mark.parametrize(
    ("family", "y"),
    [
        (Gaussian(), jnp.array([0.1, 1.0, 2.2, 2.9, 3.8])),
        (Poisson(), jnp.array([0.0, 1.0, 1.0, 2.0, 3.0])),
        (Binomial(), jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])),
        (NegativeBinomial(), jnp.array([0.0, 1.0, 2.0, 1.0, 4.0])),
    ],
)
def test_predict_generates_stable_shape_for_supported_families(family, y) -> None:
    model = glmax.GLM(family=family)
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), y=y)
    fit_result = glmax.fit(model, data)

    pred1 = glmax.predict(model, fit_result.params, data)
    pred2 = glmax.predict(model, fit_result.params, data)

    assert pred1.shape == y.shape
    assert jnp.all(jnp.isfinite(pred1))
    assert jnp.allclose(pred1, pred2)

    if isinstance(family, Binomial):
        assert jnp.all(pred1 >= 0.0)
        assert jnp.all(pred1 <= 1.0)
    elif isinstance(family, (Poisson, NegativeBinomial)):
        assert jnp.all(pred1 > 0.0)


def test_predict_boundary_rejects_invalid_nouns() -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))
    params = Params(beta=jnp.array([1.0]), disp=jnp.array(0.0))

    with pytest.raises(TypeError, match="GLM"):
        glmax.predict(object(), params, data)

    with pytest.raises(TypeError, match="Params"):
        glmax.predict(model, jnp.array([1.0]), data)

    with pytest.raises(TypeError, match="GLMData"):
        glmax.predict(model, params, jnp.array([[0.0], [1.0], [2.0]]))


def test_predict_rejects_beta_shape_mismatch() -> None:
    model = glmax.GLM(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))
    bad_params = Params(beta=jnp.array([1.0, 2.0]), disp=jnp.array(0.0))

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.predict(model, bad_params, data)
