# pattern: Imperative Shell

import pytest

import jax.numpy as jnp

import glmax

from glmax import GLM, GLMData, Params
from glmax.family import Binomial, Gaussian, NegativeBinomial, Poisson


def test_glmdata_accepts_valid_inputs_and_canonicalizes_optional_vectors() -> None:
    data = GLMData(X=jnp.ones((4, 2)), y=jnp.array([1.0, 0.0, 1.0, 2.0]))
    assert data.X.shape == (4, 2)
    assert data.y.shape == (4,)

    with_optional = GLMData(
        X=jnp.ones((4, 2)),
        y=jnp.array([1.0, 0.0, 1.0, 2.0]),
        offset=1.5,
        weights=2.0,
    )

    assert with_optional.offset is not None
    assert with_optional.weights is not None
    assert with_optional.offset.shape == (4,)
    assert with_optional.weights.shape == (4,)


def test_glmdata_rejects_non_numeric_values() -> None:
    with pytest.raises(TypeError):
        GLMData(X=[["a", "b"], ["c", "d"]], y=jnp.ones(2))

    with pytest.raises(TypeError):
        GLMData(X=jnp.ones((2, 2)), y=["a", "b"])


def test_glmdata_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError):
        GLMData(X=jnp.array([[1.0, jnp.inf], [0.0, 1.0]]), y=jnp.ones(2))

    with pytest.raises(ValueError):
        GLMData(X=jnp.ones((2, 2)), y=jnp.array([1.0, jnp.nan]))


def test_glmdata_rejects_shape_mismatches() -> None:
    with pytest.raises(ValueError):
        GLMData(X=jnp.ones(4), y=jnp.ones(4))

    with pytest.raises(ValueError):
        GLMData(X=jnp.ones((4, 2)), y=jnp.ones((4, 1)))

    with pytest.raises(ValueError):
        GLMData(X=jnp.ones((4, 2)), y=jnp.ones(3))

    with pytest.raises(ValueError):
        GLMData(X=jnp.ones((4, 2)), y=jnp.ones(4), offset=jnp.ones((4, 1)))

    with pytest.raises(ValueError):
        GLMData(X=jnp.ones((4, 2)), y=jnp.ones(4), weights=jnp.ones(3))


def test_glm_fit_accepts_glmdata_noun() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.1, 1.0, 2.0, 2.9]))
    fit_result = glmax.fit(GLM(family=Gaussian()), data)
    current_fitted_glm_type = __import__("glmax._fit", fromlist=["FittedGLM"]).FittedGLM
    assert isinstance(fit_result, current_fitted_glm_type)
    assert fit_result.params.beta.shape == (1,)


def test_top_level_fit_rejects_raw_x_y_inputs() -> None:
    with pytest.raises(TypeError, match="GLMData"):
        glmax.fit(GLM(family=Gaussian()), jnp.ones((4, 1)))


def test_glmax_fit_accepts_params_init_for_nb_without_aux() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.0, 1.0, 1.0, 2.0]))
    model = GLM(family=NegativeBinomial())
    init = Params(beta=jnp.zeros(1), disp=jnp.array(0.4), aux=None)

    fit_result = glmax.fit(model, data, init=init)

    assert list(fit_result.params._fields) == ["beta", "disp", "aux"]
    assert fit_result.params.beta.shape == (1,)
    assert jnp.allclose(fit_result.params.disp, jnp.array(1.0))
    assert fit_result.params.aux is not None
    assert float(jnp.asarray(fit_result.params.aux)) > 0.0


def test_params_schema_is_beta_disp_and_aux() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.1, 1.0, 2.0, 2.9]))
    fit_result = glmax.fit(GLM(family=Gaussian()), data)

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

    fit_result = glmax.fit(GLM(family=family), GLMData(X=X, y=y))
    if isinstance(family, Gaussian):
        assert fit_result.params.disp > 0
    else:
        assert jnp.allclose(fit_result.params.disp, jnp.array(1.0))
