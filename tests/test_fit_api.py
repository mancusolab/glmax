import numpy as np
import pytest

import equinox as eqx
import jax.numpy as jnp

import glmax
import glmax.infer as infer


def _basic_data():
    X = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    y = jnp.array([1.0, 2.0, 3.0, 4.0])
    return X, y


class _SentinelFitter(glmax.AbstractFitter):
    def __call__(
        self,
        model,
        X,
        y,
        offset,
        *,
        init=None,
        options=None,
    ):
        state = model.fit(X, y, offset_eta=offset, init=init, **(options or {}))
        return state._replace(num_iters=jnp.asarray(-1))


class _IdentityCovariance(infer.AbstractStdErrEstimator):
    def __call__(
        self,
        family,
        X,
        y,
        eta,
        mu,
        weight,
        alpha=0.0,
    ):
        del family, y, eta, mu, weight, alpha
        return jnp.eye(X.shape[1])


class _ZeroPValueTestHook(infer.AbstractHypothesisTest):
    def __call__(self, statistic, df, family):
        del df, family
        return jnp.zeros_like(statistic)


def test_gx_fit_returns_glmstate_for_gaussian():
    X, y = _basic_data()

    state = glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)

    assert isinstance(state, glmax.GLMState)


def test_gx_fit_accepts_optional_offset():
    X, y = _basic_data()
    offset = jnp.array([0.0, 0.1, -0.1, 0.0])

    state = glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, offset=offset)

    assert isinstance(state, glmax.GLMState)


def test_gx_fit_smoke_for_poisson_defaults():
    X = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.5],
            [1.0, 1.5],
        ]
    )
    y = jnp.array([1.0, 2.0, 2.0, 3.0])

    state = glmax.fit(glmax.GLM(family=glmax.Poisson()), X, y)

    assert isinstance(state, glmax.GLMState)


def test_gx_fit_accepts_custom_fitter_strategy():
    X, y = _basic_data()

    state = glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, fitter=_SentinelFitter())

    assert state.num_iters == -1


def test_gx_fit_accepts_solver_strategy_swap():
    X, y = _basic_data()

    state = glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, solver=glmax.QRSolver())

    assert isinstance(state, glmax.GLMState)


def test_gx_fit_accepts_covariance_strategy_swap():
    X, y = _basic_data()

    state = glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, covariance=_IdentityCovariance())

    assert jnp.allclose(state.se, jnp.ones_like(state.se))


def test_infer_state_contracts_are_exported():
    assert glmax.GLMState is infer.GLMState
    assert "beta" in infer.IRLSState._fields


def test_gx_fit_default_pipeline_exposes_complete_fields():
    X, y = _basic_data()

    state = glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)

    assert state.beta.shape == (X.shape[1],)
    assert state.se.shape == (X.shape[1],)
    assert state.p.shape == (X.shape[1],)
    assert state.eta.shape == y.shape
    assert state.mu.shape == y.shape
    assert state.num_iters >= 0
    assert bool(state.converged) in (True, False)


def test_gx_fit_accepts_custom_hypothesis_test_hook():
    X, y = _basic_data()

    state = glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, tests=_ZeroPValueTestHook())

    assert jnp.allclose(state.p, jnp.zeros_like(state.p))


def test_gx_fit_rejects_non_2d_X():
    _, y = _basic_data()
    X = jnp.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="X must be a 2D array"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)


def test_gx_fit_rejects_non_1d_y():
    X, y = _basic_data()
    y = y[:, None]

    with pytest.raises(ValueError, match="y must be a 1D array"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)


def test_gx_fit_rejects_mismatched_sample_count():
    X, y = _basic_data()
    y = y[:3]

    with pytest.raises(ValueError, match="X and y must have the same number of rows"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)


def test_gx_fit_rejects_mismatched_offset_length():
    X, y = _basic_data()
    offset = jnp.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="offset must have length equal to the number of rows in X"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, offset=offset)


def test_gx_fit_rejects_non_finite_X():
    X, y = _basic_data()
    X = X.at[0, 0].set(jnp.inf)

    with pytest.raises(ValueError, match="X must contain only finite values"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)


def test_gx_fit_rejects_non_finite_y():
    X, y = _basic_data()
    y = y.at[0].set(jnp.nan)

    with pytest.raises(ValueError, match="y must contain only finite values"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)


def test_gx_fit_rejects_non_finite_offset():
    X, y = _basic_data()
    offset = jnp.array([0.0, 0.0, 0.0, jnp.nan])

    with pytest.raises(ValueError, match="offset must contain only finite values"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, offset=offset)


def test_gx_fit_rejects_non_numeric_X():
    X, y = _basic_data()
    X = np.array([["1.0", "0.0"], ["1.0", "1.0"], ["1.0", "2.0"], ["1.0", "3.0"]], dtype=str)

    with pytest.raises(TypeError, match="X must have a numeric dtype"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)


def test_gx_fit_rejects_non_numeric_y():
    X, y = _basic_data()
    y = np.array(["1.0", "2.0", "3.0", "4.0"], dtype=str)

    with pytest.raises(TypeError, match="y must have a numeric dtype"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y)


def test_gx_fit_rejects_non_numeric_offset():
    X, y = _basic_data()
    offset = np.array(["0.0", "0.0", "0.0", "0.0"], dtype=str)

    with pytest.raises(TypeError, match="offset must have a numeric dtype"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, offset=offset)


def test_gx_fit_rejects_invalid_family_link_pair():
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Poisson())
    model = eqx.tree_at(lambda m: m.family.glink, model, glmax.Logit())

    with pytest.raises(ValueError, match="Invalid family/link combination"):
        glmax.fit(model, X, y)


def test_gx_fit_rejects_invalid_solver_strategy():
    X, y = _basic_data()

    with pytest.raises(TypeError, match="solver must implement AbstractLinearSolver"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, solver=object())


def test_gx_fit_rejects_invalid_covariance_strategy():
    X, y = _basic_data()

    with pytest.raises(TypeError, match="covariance must implement AbstractStdErrEstimator"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, covariance=object())


def test_gx_fit_rejects_unsupported_fitter_strategy():
    X, y = _basic_data()

    with pytest.raises(TypeError, match="fitter must implement AbstractFitter"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, fitter=object())


def test_gx_fit_rejects_unsupported_tests_strategy():
    X, y = _basic_data()

    with pytest.raises(TypeError, match="tests must implement AbstractHypothesisTest"):
        glmax.fit(glmax.GLM(family=glmax.Gaussian()), X, y, tests=object())
