# pattern: Imperative Shell


import pytest

import jax.numpy as jnp

import glmax

from glmax import GLMData, Params
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
    model = glmax.specify(family=family)
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), y=y)
    fit_result = glmax.fit(model, data)

    pred1 = glmax.predict(model, fit_result.params, data)
    pred2 = glmax.predict(model, fit_result.params, data)

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
    ],
)
def test_predict_ignores_aux_for_families_without_aux_state(family, X, y) -> None:
    model = glmax.specify(family=family)
    data = GLMData(X=X, y=y)
    fit_result = glmax.fit(model, data)
    assert fit_result.params._fields == ("beta", "disp", "aux")
    assert fit_result.params.aux is None
    if isinstance(family, Gaussian):
        assert float(jnp.asarray(fit_result.params.disp)) > 0.0
    else:
        assert jnp.allclose(fit_result.params.disp, jnp.array(1.0))
    params_with_aux = Params(beta=fit_result.params.beta, disp=fit_result.params.disp, aux=jnp.array(0.25))

    assert jnp.allclose(
        glmax.predict(model, params_with_aux, data),
        glmax.predict(model, fit_result.params, data),
    )


def test_predict_boundary_rejects_invalid_nouns() -> None:
    model = glmax.specify(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))
    params = Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux=jnp.array(0.2))

    with pytest.raises(TypeError, match="GLM"):
        glmax.predict(object(), params, data)

    with pytest.raises(TypeError, match="Params"):
        glmax.predict(model, jnp.array([1.0]), data)

    with pytest.raises(TypeError, match="GLMData"):
        glmax.predict(model, params, jnp.array([[0.0], [1.0], [2.0]]))


def test_predict_rejects_beta_shape_mismatch() -> None:
    model = glmax.specify(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))
    bad_params = Params(beta=jnp.array([1.0, 2.0]), disp=jnp.array(0.0), aux=None)

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.predict(model, bad_params, data)


def test_predict_rejects_non_numeric_or_non_scalar_params() -> None:
    model = glmax.specify(family=Gaussian())
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]))

    with pytest.raises(TypeError, match="Params.beta must be numeric"):
        glmax.predict(model, Params(beta=["bad"], disp=jnp.array(0.0), aux=None), data)

    with pytest.raises(TypeError, match="Params.disp must be numeric"):
        glmax.predict(model, Params(beta=jnp.array([1.0]), disp="bad", aux=None), data)

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
