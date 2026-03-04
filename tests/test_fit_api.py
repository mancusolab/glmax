import importlib
import warnings

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


def _state_parity_fields():
    return (
        "beta",
        "se",
        "p",
        "eta",
        "mu",
        "glm_wt",
        "num_iters",
        "converged",
        "infor_inv",
        "resid",
        "alpha",
    )


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


class _OptionsAwareFitter(glmax.AbstractFitter):
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
        options = {} if options is None else dict(options)
        se_estimator = options.pop("se_estimator", infer.FisherInfoError())
        test_hook = options.pop("test_hook", None)
        state = model.fit(X, y, offset_eta=offset, init=init, se_estimator=se_estimator, **options)
        if test_hook is not None:
            p = test_hook(state.z, X.shape[0] - X.shape[1], model.family)
            state = state._replace(p=p)
        return state


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


@pytest.mark.parametrize(
    "family",
    (
        glmax.Gaussian(),
        glmax.Poisson(),
        glmax.Binomial(),
        glmax.NegativeBinomial(),
    ),
)
def test_wrapper_and_direct_entrypoints_have_output_parity(family):
    X = jnp.array(
        [
            [1.0, 0.1],
            [1.0, 0.3],
            [1.0, 0.7],
            [1.0, 1.1],
            [1.0, 1.7],
            [1.0, 2.3],
        ]
    )
    if isinstance(family, glmax.Binomial):
        y = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    elif isinstance(family, glmax.NegativeBinomial):
        y = jnp.array([1.0, 1.0, 2.0, 3.0, 6.0, 8.0])
    elif isinstance(family, glmax.Poisson):
        y = jnp.array([1.0, 1.0, 2.0, 3.0, 4.0, 6.0])
    else:
        y = jnp.array([1.0, 1.4, 2.0, 2.8, 3.2, 4.1])

    model = glmax.GLM(family=family)
    direct_state = glmax.fit(model, X, y)
    wrapped_state = model.fit(X, y)

    for field in _state_parity_fields():
        direct_value = getattr(direct_state, field)
        wrapped_value = getattr(wrapped_state, field)
        assert jnp.allclose(direct_value, wrapped_value)


def test_glm_fit_delegates_to_package_fit(monkeypatch):
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Gaussian())
    expected = glmax.fit(model, X, y)
    fit_module = importlib.import_module("glmax.fit")
    calls = {"count": 0}

    def _fake_fit(*args, **kwargs):
        calls["count"] += 1
        assert args[0] is model
        return expected

    monkeypatch.setattr(fit_module, "fit", _fake_fit)

    state = model.fit(X, y)
    assert calls["count"] == 1
    assert state == expected


def test_wrapper_successfully_maps_offset_and_covariance_parameters():
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Gaussian())
    offset = jnp.array([0.1, 0.0, -0.1, 0.2])
    covariance = _IdentityCovariance()

    direct_state = glmax.fit(model, X, y, offset=offset, covariance=covariance)
    wrapped_state = model.fit(X, y, offset_eta=offset, se_estimator=covariance)

    for field in _state_parity_fields():
        assert jnp.allclose(getattr(direct_state, field), getattr(wrapped_state, field))


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


def test_gx_fit_forwards_covariance_to_custom_fitter():
    X, y = _basic_data()

    state = glmax.fit(
        glmax.GLM(family=glmax.Gaussian()),
        X,
        y,
        fitter=_OptionsAwareFitter(),
        covariance=_IdentityCovariance(),
    )

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


def test_gx_fit_forwards_test_hook_to_custom_fitter():
    X, y = _basic_data()

    state = glmax.fit(
        glmax.GLM(family=glmax.Gaussian()),
        X,
        y,
        fitter=_OptionsAwareFitter(),
        tests=_ZeroPValueTestHook(),
    )

    assert jnp.allclose(state.p, jnp.zeros_like(state.p))


def test_gx_fit_matches_glm_fit_convergence_metadata():
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Gaussian())

    gx_state = glmax.fit(model, X, y)
    glm_state = model.fit(X, y)

    assert gx_state.num_iters == glm_state.num_iters
    assert bool(gx_state.converged) == bool(glm_state.converged)


def test_gx_fit_matches_glm_fit_nb_with_non_default_max_iter():
    X = jnp.array(
        [
            [1.0, 0.1],
            [1.0, 0.2],
            [1.0, 0.4],
            [1.0, 0.8],
            [1.0, 1.6],
            [1.0, 3.2],
        ]
    )
    y = jnp.array([1.0, 1.0, 2.0, 3.0, 6.0, 8.0])
    model = glmax.GLM(family=glmax.NegativeBinomial())

    gx_state = glmax.fit(model, X, y, options={"max_iter": 7})
    glm_state = model.fit(X, y, max_iter=7)

    assert jnp.allclose(gx_state.beta, glm_state.beta)
    assert gx_state.num_iters == glm_state.num_iters
    assert bool(gx_state.converged) == bool(glm_state.converged)


def test_wrapper_and_direct_entrypoints_shape_failure_parity():
    model = glmax.GLM(family=glmax.Gaussian())
    X = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError) as direct_error:
        glmax.fit(model, X, y)
    with pytest.raises(ValueError) as wrapped_error:
        model.fit(X, y)

    assert isinstance(wrapped_error.value, type(direct_error.value))
    assert str(direct_error.value) == str(wrapped_error.value)


def test_wrapper_and_direct_entrypoints_invalid_runtime_parity():
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Poisson())
    model = eqx.tree_at(lambda m: m.family.glink, model, glmax.Logit())

    with pytest.raises(ValueError) as direct_error:
        glmax.fit(model, X, y)
    with pytest.raises(ValueError) as wrapped_error:
        model.fit(X, y)

    assert isinstance(wrapped_error.value, type(direct_error.value))
    assert str(direct_error.value) == str(wrapped_error.value)


def test_wrapper_and_direct_entrypoints_strategy_failure_parity():
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Gaussian())

    with pytest.raises(TypeError) as direct_error:
        glmax.fit(model, X, y, covariance=object())
    with pytest.raises(TypeError) as wrapped_error:
        model.fit(X, y, se_estimator=object())

    assert isinstance(wrapped_error.value, type(direct_error.value))
    assert str(direct_error.value) == str(wrapped_error.value)


def test_wrapper_and_direct_entrypoints_offset_failure_parity():
    X, y = _basic_data()
    offset = jnp.array([0.0, 0.0, 0.0])
    model = glmax.GLM(family=glmax.Gaussian())

    with pytest.raises(ValueError) as direct_error:
        glmax.fit(model, X, y, offset=offset)
    with pytest.raises(ValueError) as wrapped_error:
        model.fit(X, y, offset_eta=offset)

    assert isinstance(wrapped_error.value, type(direct_error.value))
    assert str(direct_error.value) == str(wrapped_error.value)


def test_wrapper_and_direct_entrypoints_invalid_scalar_offset_parity():
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Gaussian())

    with pytest.raises(TypeError) as direct_error:
        glmax.fit(model, X, y, offset="invalid")
    with pytest.raises(TypeError) as wrapped_error:
        model.fit(X, y, offset_eta="invalid")

    assert isinstance(wrapped_error.value, type(direct_error.value))
    assert str(direct_error.value) == str(wrapped_error.value)


def test_glm_fit_emits_no_warning_by_default(monkeypatch):
    monkeypatch.delenv("GLMAX_WARN_GLM_FIT_COMPAT", raising=False)
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Gaussian())

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(X, y)

    assert captured == []


def test_glm_fit_emits_opt_in_compat_warning(monkeypatch):
    monkeypatch.setenv("GLMAX_WARN_GLM_FIT_COMPAT", "1")
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Gaussian())

    with pytest.warns(UserWarning, match="GLM.fit is a compatibility wrapper over glmax.fit"):
        model.fit(X, y)


def test_glm_fit_emits_opt_in_compat_warning_for_true(monkeypatch):
    monkeypatch.setenv("GLMAX_WARN_GLM_FIT_COMPAT", "true")
    X, y = _basic_data()
    model = glmax.GLM(family=glmax.Gaussian())

    with pytest.warns(UserWarning, match="GLM.fit is a compatibility wrapper over glmax.fit"):
        model.fit(X, y)


def test_glm_fit_emits_single_warning_for_nb_call(monkeypatch):
    monkeypatch.setenv("GLMAX_WARN_GLM_FIT_COMPAT", "1")
    X = jnp.array(
        [
            [1.0, 0.1],
            [1.0, 0.2],
            [1.0, 0.4],
            [1.0, 0.8],
        ]
    )
    y = jnp.array([1.0, 1.0, 2.0, 3.0])
    model = glmax.GLM(family=glmax.NegativeBinomial())

    with pytest.warns(UserWarning, match="GLM.fit is a compatibility wrapper over glmax.fit") as record:
        model.fit(X, y)

    assert len(record) == 1


def test_wrapper_and_direct_hypothesis_method_parity():
    model = glmax.GLM(family=glmax.Gaussian())
    statistic = jnp.array([0.7, 1.2, -0.3])
    df = 10

    wrapper_p = model.wald_test(statistic, df)
    direct_hook_p = infer.WaldTest()(statistic, df, model.family)

    assert jnp.allclose(wrapper_p, direct_hook_p)


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
