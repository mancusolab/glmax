# pattern: Imperative Shell

import pytest

import jax.numpy as jnp

import glmax

from glmax import Params
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


def test_glm_fit_accepts_x_y_inputs() -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    y = jnp.array([0.1, 1.0, 2.0, 2.9])
    fit_result = glmax.fit(Gaussian(), X, y)
    current_fitted_glm_type = __import__("glmax._fit", fromlist=["FittedGLM"]).FittedGLM
    assert isinstance(fit_result, current_fitted_glm_type)
    assert fit_result.params.beta.shape == (1,)


def test_glmax_fit_accepts_params_init_for_nb_without_aux() -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    y = jnp.array([0.0, 1.0, 1.0, 2.0])
    init = Params(beta=jnp.zeros(1), disp=jnp.array(0.4), aux=None)

    fit_result = glmax.fit(NegativeBinomial(), X, y, init=init)

    assert list(fit_result.params._fields) == ["beta", "disp", "aux"]
    assert fit_result.params.beta.shape == (1,)
    assert jnp.allclose(fit_result.params.disp, jnp.array(1.0))
    assert fit_result.params.aux is not None
    assert float(jnp.asarray(fit_result.params.aux)) > 0.0


def test_params_schema_is_beta_disp_and_aux() -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    y = jnp.array([0.1, 1.0, 2.0, 2.9])
    fit_result = glmax.fit(Gaussian(), X, y)

    assert list(fit_result.params._fields) == ["beta", "disp", "aux"]
    assert fit_result.params.aux is None
    assert not hasattr(fit_result, "alpha")


@pytest.mark.parametrize("family", [Gaussian(), Poisson(), Binomial()])
def test_fixed_dispersion_families_emit_deterministic_disp(family) -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    if isinstance(family, Binomial):
        y = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    else:
        y = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    fit_result = glmax.fit(family, X, y)
    if isinstance(family, Gaussian):
        assert fit_result.params.disp > 0
    else:
        assert jnp.allclose(fit_result.params.disp, jnp.array(1.0))
