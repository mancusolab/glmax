# pattern: Imperative Shell


import pytest

import jax.numpy as jnp

import glmax

from glmax import Params
from glmax.family import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson


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
    X = jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    fit_result = glmax.fit(family, X, y)

    pred1 = glmax.predict(family, fit_result.params, X)
    pred2 = glmax.predict(family, fit_result.params, X)

    assert pred1.shape == y.shape
    assert jnp.all(jnp.isfinite(pred1))
    assert jnp.allclose(pred1, pred2)
    assert fit_result.params._fields == ("beta", "disp", "aux")

    if isinstance(family, NegativeBinomial):
        assert jnp.allclose(fit_result.params.disp, jnp.array(1.0))
        assert fit_result.params.aux is not None
        assert float(jnp.asarray(fit_result.params.aux)) > 0.0
    else:
        assert fit_result.params.aux is None

    if isinstance(family, Binomial):
        assert jnp.all(pred1 >= 0.0)
        assert jnp.all(pred1 <= 1.0)
    elif isinstance(family, (Poisson, NegativeBinomial)):
        assert jnp.all(pred1 > 0.0)


@pytest.mark.parametrize(
    ("family", "X", "y"),
    [
        (Gaussian(), jnp.array([[0.0], [1.0], [2.0], [3.0]]), jnp.array([0.1, 1.0, 2.2, 2.9])),
        (Gamma(), jnp.array([[1.0], [2.0], [3.0], [4.0]]), jnp.array([0.8, 1.1, 1.7, 2.4])),
        (Poisson(), jnp.array([[0.0], [1.0], [2.0], [3.0]]), jnp.array([0.0, 1.0, 1.0, 2.0])),
        (Binomial(), jnp.array([[0.0], [1.0], [2.0], [3.0]]), jnp.array([0.0, 0.0, 1.0, 1.0])),
    ],
)
def test_predict_ignores_aux_for_families_without_aux_state(family, X, y) -> None:
    fit_result = glmax.fit(family, X, y)
    assert fit_result.params._fields == ("beta", "disp", "aux")
    assert fit_result.params.aux is None
    if isinstance(family, Gaussian):
        assert float(jnp.asarray(fit_result.params.disp)) > 0.0
    else:
        assert jnp.allclose(fit_result.params.disp, jnp.array(1.0))
    params_with_aux = Params(beta=fit_result.params.beta, disp=fit_result.params.disp, aux=jnp.array(0.25))

    assert jnp.allclose(
        glmax.predict(family, params_with_aux, X),
        glmax.predict(family, fit_result.params, X),
    )


def test_predict_boundary_rejects_invalid_nouns() -> None:
    family = Gaussian()
    X = jnp.array([[0.0], [1.0], [2.0]])
    params = Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux=jnp.array(0.2))

    with pytest.raises(TypeError, match="ExponentialDispersionFamily"):
        glmax.predict(object(), params, X)

    with pytest.raises(TypeError, match="Params"):
        glmax.predict(family, jnp.array([1.0]), X)
