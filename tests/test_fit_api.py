import importlib
import inspect

from typing import Tuple

import numpy as np
import pytest

from utils import assert_array_eq, assert_glm_state_parity

import jax.nn
import jax.numpy as jnp
import jax.random as rdm

import glmax


def test_package_root_fit_export_identity():
    from glmax import fit as package_fit

    assert callable(glmax.fit)
    assert package_fit is glmax.fit


def simulate_glm_data(
    key: rdm.PRNGKey,
    n_samples: int = 180,
    n_features: int = 5,
    family: str = "poisson",
    dispersion: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    key, x_key, beta_key, noise_key, extra_key = rdm.split(key, 5)

    X = rdm.normal(x_key, shape=(n_samples, n_features))
    X = X - X.mean(axis=0) / (X.std(axis=0))
    beta_true = rdm.normal(beta_key, shape=(n_features,))
    eta = X @ beta_true

    if family == "poisson":
        y = rdm.poisson(noise_key, jnp.exp(eta))
    elif family == "normal":
        y = eta + rdm.normal(noise_key, shape=(n_samples,))
    elif family == "binomial":
        p = jnp.clip(jax.nn.sigmoid(eta), 1e-5, 1 - 1e-5)
        y = rdm.bernoulli(noise_key, p).astype(jnp.int32)
    elif family == "negative_binomial":
        lam = jnp.exp(eta)
        r = jnp.array(1.0 / dispersion)
        gamma_sample = rdm.gamma(noise_key, r, shape=lam.shape)
        y = rdm.poisson(extra_key, lam=gamma_sample * lam / r)
    else:
        raise ValueError(f"Unsupported family: {family}")

    return X, y


def _assert_boundary_error_parity(
    X,
    y,
    *,
    offset_eta=0.0,
    init=None,
    alpha_init=None,
    exc_type=ValueError,
    message="",
):
    with pytest.raises(exc_type, match=message):
        glmax.fit(
            X,
            y,
            family=glmax.Poisson(),
            solver=glmax.CholeskySolver(),
            offset_eta=offset_eta,
            init=init,
            alpha_init=alpha_init,
        )

    model = glmax.GLM(family=glmax.Poisson(), solver=glmax.CholeskySolver())
    with pytest.raises(exc_type, match=message):
        model.fit(X, y, offset_eta=offset_eta, init=init, alpha_init=alpha_init)


def test_public_entrypoint_docstrings_follow_contract_sections():
    optimize = importlib.import_module("glmax.infer.optimize")
    entrypoints = (glmax.fit, glmax.GLM.fit, optimize.irls)

    for fn in entrypoints:
        doc = inspect.getdoc(fn)
        assert doc is not None
        assert "**Arguments:**" in doc
        assert "**Returns:**" in doc
        assert "**Raises:**" in doc or "**Failure Modes:**" in doc


@pytest.mark.parametrize(
    ("family_name", "family", "fit_kwargs"),
    [
        ("normal", glmax.Gaussian(), {}),
        ("poisson", glmax.Poisson(), {}),
        ("binomial", glmax.Binomial(), {}),
        ("negative_binomial", glmax.NegativeBinomial(), {"tol": 1e-8}),
    ],
)
def test_wrapper_and_canonical_fit_parity(getkey, family_name, family, fit_kwargs):
    from glmax import fit as package_fit

    X, y = simulate_glm_data(getkey(), family=family_name, dispersion=2.0)
    solver = glmax.CholeskySolver()

    direct_state = package_fit(X, y, family=family, solver=solver, **fit_kwargs)
    wrapper_state = glmax.GLM(family=family, solver=solver).fit(X, y, **fit_kwargs)

    assert_glm_state_parity(wrapper_state, direct_state, rtol=1e-7, atol=1e-8)


def test_canonical_fit_routes_pvalues_through_inference_strategy(monkeypatch, getkey):
    fit_module = importlib.import_module("glmax.fit")

    def fake_wald_test(statistic, df, family):
        del df, family
        return jnp.full_like(statistic, 0.2222)

    monkeypatch.setattr(fit_module, "inference_wald_test", fake_wald_test, raising=False)

    X, y = simulate_glm_data(getkey(), family="poisson")
    state = glmax.fit(X, y, family=glmax.Poisson(), solver=glmax.CholeskySolver())

    assert_array_eq(state.p, jnp.full_like(state.p, 0.2222), rtol=0.0, atol=1e-12)


def test_custom_fitter_strategy_injection_for_canonical_and_glm(getkey):
    fitters = importlib.import_module("glmax.infer.fitters")
    calls = {"count": 0}

    class CustomFitter(fitters.AbstractGLMFitter):
        def __call__(
            self,
            X,
            y,
            family,
            solver,
            eta,
            max_iter=1000,
            tol=1e-3,
            step_size=1.0,
            offset_eta=0.0,
            alpha_init=0.0,
        ):
            del y, family, solver, max_iter, tol, step_size, offset_eta, alpha_init
            calls["count"] += 1
            beta = jnp.zeros((X.shape[1],), dtype=eta.dtype)
            return fitters.IRLSState(beta=beta, num_iters=3, converged=jnp.asarray(True), alpha=jnp.asarray(0.0))

    X, y = simulate_glm_data(getkey(), family="poisson")
    custom = CustomFitter()

    state_direct = glmax.fit(X, y, family=glmax.Poisson(), solver=glmax.CholeskySolver(), fitter=custom)
    model = glmax.GLM(family=glmax.Poisson(), solver=glmax.CholeskySolver(), fitter=custom)
    state_wrapper = model.fit(X, y)

    assert calls["count"] == 2
    assert int(state_direct.num_iters) == 3
    assert int(state_wrapper.num_iters) == 3


def test_equivalent_custom_fitter_preserves_regression_parity(getkey):
    fitters = importlib.import_module("glmax.infer.fitters")

    class DelegatingFitter(fitters.AbstractGLMFitter):
        base: fitters.AbstractGLMFitter = fitters.IRLSFitter()

        def __call__(
            self,
            X,
            y,
            family,
            solver,
            eta,
            max_iter=1000,
            tol=1e-3,
            step_size=1.0,
            offset_eta=0.0,
            alpha_init=0.0,
        ):
            return self.base(
                X,
                y,
                family,
                solver,
                eta,
                max_iter=max_iter,
                tol=tol,
                step_size=step_size,
                offset_eta=offset_eta,
                alpha_init=alpha_init,
            )

    X, y = simulate_glm_data(getkey(), family="poisson")
    solver = glmax.CholeskySolver()
    custom_fitter = DelegatingFitter()

    baseline = glmax.fit(X, y, family=glmax.Poisson(), solver=solver)
    injected = glmax.fit(X, y, family=glmax.Poisson(), solver=solver, fitter=custom_fitter)
    wrapper_injected = glmax.GLM(family=glmax.Poisson(), solver=solver, fitter=custom_fitter).fit(X, y)

    assert_glm_state_parity(injected, baseline, rtol=1e-7, atol=1e-8)
    assert_glm_state_parity(wrapper_injected, baseline, rtol=1e-7, atol=1e-8)


def test_wrapper_and_canonical_share_boundary_normalization(monkeypatch, getkey):
    fit_module = importlib.import_module("glmax.fit")
    calls = {"count": 0}

    original = fit_module._normalize_fit_inputs

    def spy_normalize(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fit_module, "_normalize_fit_inputs", spy_normalize)

    X, y = simulate_glm_data(getkey(), family="poisson")
    glmax.fit(X, y, family=glmax.Poisson(), solver=glmax.CholeskySolver())
    glmax.GLM(family=glmax.Poisson(), solver=glmax.CholeskySolver()).fit(X, y)

    assert calls["count"] >= 2


def test_boundary_rejects_non_numeric_dtypes_with_deterministic_errors():
    X = np.array([["a", "b"], ["c", "d"]])
    y = jnp.array([1.0, 0.0])
    _assert_boundary_error_parity(X, y, exc_type=TypeError, message="X must have a numeric dtype")

    X = jnp.array([[1.0, 0.0], [1.0, 1.0]])
    y = np.array(["0", "1"])
    _assert_boundary_error_parity(X, y, exc_type=TypeError, message="y must have a numeric dtype")

    offset = np.array(["x", "y"])
    _assert_boundary_error_parity(
        X, jnp.array([0.0, 1.0]), offset_eta=offset, exc_type=TypeError, message="offset_eta must have a numeric dtype"
    )


def test_boundary_rejects_invalid_rank_shape_and_finiteness():
    _assert_boundary_error_parity(
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([1.0, 2.0, 3.0]),
        exc_type=ValueError,
        message="X must be a 2D array",
    )
    _assert_boundary_error_parity(
        jnp.array([[1.0, 0.0], [1.0, 1.0]]),
        jnp.array([1.0]),
        exc_type=ValueError,
        message="X and y must have the same number of rows",
    )
    _assert_boundary_error_parity(
        jnp.array([[1.0, 0.0], [1.0, 1.0]]),
        jnp.array([1.0, 0.0]),
        offset_eta=jnp.array([0.0]),
        exc_type=ValueError,
        message="offset_eta must be a scalar or a 1D array with length equal to the number of rows in X",
    )
    _assert_boundary_error_parity(
        jnp.array([[1.0, 0.0], [1.0, 1.0]]),
        jnp.array([1.0, 0.0]),
        init=jnp.array([0.0]),
        exc_type=ValueError,
        message="init must be a 1D array with length equal to the number of rows in X",
    )
    _assert_boundary_error_parity(
        jnp.array([[1.0, jnp.nan], [1.0, 1.0]]),
        jnp.array([1.0, 0.0]),
        exc_type=ValueError,
        message="X must contain only finite values",
    )
    _assert_boundary_error_parity(
        jnp.array([[1.0, 0.0], [1.0, 1.0]]),
        jnp.array([1.0, jnp.inf]),
        exc_type=ValueError,
        message="y must contain only finite values",
    )


def test_valid_input_parity_is_preserved_after_boundary_cleanup(getkey):
    X, y = simulate_glm_data(getkey(), family="poisson")
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    offset = 0.25
    solver = glmax.CholeskySolver()

    direct_state = glmax.fit(X_np, y_np, family=glmax.Poisson(), solver=solver, offset_eta=offset)
    wrapper_state = glmax.GLM(family=glmax.Poisson(), solver=solver).fit(X_np, y_np, offset_eta=offset)

    assert_glm_state_parity(wrapper_state, direct_state, rtol=1e-7, atol=1e-8)
