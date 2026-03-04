import jax.numpy as jnp

import glmax


def test_gx_fit_returns_glmstate_for_gaussian():
    X = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    y = jnp.array([1.0, 2.0, 3.0, 4.0])

    state = glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)

    assert isinstance(state, glmax.GLMState)
