from dataclasses import fields

import pytest

import jax.numpy as jnp

import glmax

from glmax import GLM, GLMData
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
        mask=True,
    )

    assert with_optional.offset is not None
    assert with_optional.weights is not None
    assert with_optional.mask is not None
    assert with_optional.offset.shape == (4,)
    assert with_optional.weights.shape == (4,)
    assert with_optional.mask.shape == (4,)
    assert with_optional.mask.dtype == jnp.bool_


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


def test_glmdata_rejects_shape_mismatches_and_malformed_mask() -> None:
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

    with pytest.raises(TypeError):
        GLMData(X=jnp.ones((4, 2)), y=jnp.ones(4), mask=jnp.ones(4))


def test_glm_fit_accepts_glmdata_noun() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.1, 1.0, 2.0, 2.9]))
    fit_result = GLM(family=Gaussian()).fit(data)
    assert fit_result.params.beta.shape == (1,)


def test_top_level_fit_rejects_raw_x_y_inputs() -> None:
    with pytest.raises(TypeError, match="GLMData"):
        glmax.fit(GLM(family=Gaussian()), jnp.ones((4, 1)))


def test_glm_fit_accepts_canonical_disp_init_keyword() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.0, 1.0, 1.0, 2.0]))
    model = GLM(family=NegativeBinomial())

    fit_result = model.fit(data, disp_init=jnp.array(0.4))

    assert fit_result.params.beta.shape == (1,)


def test_glm_fit_rejects_all_false_mask_with_deterministic_error() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0]]), y=jnp.array([0.0, 1.0, 2.0]), mask=False)

    with pytest.raises(ValueError, match="mask removes all samples"):
        GLM(family=Gaussian()).fit(data)


def test_params_schema_is_beta_and_disp_only() -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0]]), y=jnp.array([0.1, 1.0, 2.0, 2.9]))
    fit_result = GLM(family=Gaussian()).fit(data)

    assert [field.name for field in fields(fit_result.params)] == ["beta", "disp"]
    assert not hasattr(fit_result, "alpha")


@pytest.mark.parametrize("family", [Gaussian(), Poisson(), Binomial()])
def test_fixed_dispersion_families_emit_deterministic_disp(family) -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    if isinstance(family, Binomial):
        y = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    else:
        y = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    fit_result = GLM(family=family).fit(GLMData(X=X, y=y))
    assert jnp.allclose(fit_result.params.disp, jnp.array(0.0))
