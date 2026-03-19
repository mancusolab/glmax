# pattern: Imperative Shell

import importlib
import warnings

from typing import ClassVar

import numpy as np
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose
from statsmodels.genmod.families import links as sm_links
from statsmodels.tools.sm_exceptions import DomainWarning

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import glmax

from glmax import AbstractFitter, FitResult, FittedGLM, GLMData, InferenceResult, Params
from glmax._fit import IRLSFitter
from glmax._fit.solve import AbstractLinearSolver, CholeskySolver, QRSolver
from glmax.family import Binomial, Gamma, Gaussian, NegativeBinomial, Poisson
from glmax.family.dist import ExponentialDispersionFamily
from glmax.family.links import IdentityLink, InverseLink, LogitLink, LogLink, NBLink, PowerLink
from glmax.family.utils import t_cdf
from glmax.glm import specify


def _make_fit_result() -> FitResult:
    return FitResult(
        params=Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux=jnp.array(0.25)),
        X=jnp.array([[1.0], [1.0]]),
        y=jnp.array([1.0, 1.0]),
        eta=jnp.array([1.0, 1.0]),
        mu=jnp.array([1.0, 1.0]),
        glm_wt=jnp.array([1.0, 1.0]),
        converged=jnp.array(True),
        num_iters=jnp.array(1),
        objective=jnp.array(0.0),
        objective_delta=jnp.array(-1e-4),
        score_residual=jnp.array([0.0, 0.0]),
    )


class _CanonicalWarmStartFamily(ExponentialDispersionFamily):
    glink: IdentityLink = IdentityLink()
    _links: ClassVar[list[type[IdentityLink]]] = [IdentityLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def negloglikelihood(self, y, eta, disp=1.0, aux=None):
        del aux
        resid = jnp.asarray(y) - jnp.asarray(eta)
        safe_disp = jnp.maximum(jnp.asarray(disp), jnp.asarray(1.0))
        return jnp.sum(jnp.square(resid)) / safe_disp + safe_disp

    def variance(self, mu, disp=1.0, aux=None):
        del aux
        safe_disp = jnp.maximum(jnp.asarray(disp), jnp.asarray(1.0))
        return jnp.ones_like(jnp.asarray(mu)) * safe_disp

    def sample(self, key, eta, disp=1.0, aux=None):
        del key, disp, aux
        return jnp.asarray(eta)

    def update_nuisance(self, X, y, eta, disp, step_size=1.0, aux=None):
        del X, y, eta, step_size
        new_disp = jnp.maximum(jnp.asarray(disp), jnp.asarray(1.0))
        new_aux = jnp.maximum(jnp.asarray(aux), jnp.asarray(0.25)) if aux is not None else jnp.asarray(0.25)
        return new_disp, new_aux

    def init_nuisance(self):
        return jnp.asarray(1.0), jnp.asarray(0.25)


class _NonIdempotentCanonicalWarmStartFamily(ExponentialDispersionFamily):
    glink: IdentityLink = IdentityLink()
    _links: ClassVar[list[type[IdentityLink]]] = [IdentityLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def negloglikelihood(self, y, eta, disp=1.0, aux=None):
        del aux
        resid = jnp.asarray(y) - jnp.asarray(eta)
        safe_disp = jnp.asarray(disp)
        return jnp.sum(jnp.square(resid)) / safe_disp + safe_disp

    def variance(self, mu, disp=1.0, aux=None):
        del aux
        safe_disp = jnp.asarray(disp)
        return jnp.ones_like(jnp.asarray(mu)) * safe_disp

    def sample(self, key, eta, disp=1.0, aux=None):
        del key, disp, aux
        return jnp.asarray(eta)

    def update_nuisance(self, X, y, eta, disp, step_size=1.0, aux=None):
        del X, y, eta, step_size
        disp_array = jnp.asarray(disp)
        new_disp = jnp.where(disp_array < 1.0, jnp.asarray(1.0), jnp.asarray(2.0))
        aux_array = jnp.asarray(aux) if aux is not None else jnp.asarray(0.0)
        new_aux = jnp.where(aux_array < 1.0, jnp.asarray(1.0), jnp.asarray(2.0))
        return new_disp, new_aux

    def init_nuisance(self):
        return jnp.asarray(1.0), jnp.asarray(2.0)


class _AuxSensitiveIRLSFamily(ExponentialDispersionFamily):
    glink: IdentityLink = IdentityLink()
    _links: ClassVar[list[type[IdentityLink]]] = [IdentityLink]
    _bounds: ClassVar[tuple[float, float]] = (-jnp.inf, jnp.inf)

    def negloglikelihood(self, y, eta, disp=1.0, aux=None):
        del disp
        alpha = jnp.asarray(0.5) if aux is None else jnp.asarray(aux)
        resid = jnp.asarray(y) - jnp.asarray(eta)
        return jnp.sum(jnp.square(resid)) / alpha + alpha

    def variance(self, mu, disp=1.0, aux=None):
        del disp
        alpha = jnp.asarray(0.5) if aux is None else jnp.asarray(aux)
        return jnp.ones_like(jnp.asarray(mu)) * alpha

    def sample(self, key, eta, disp=1.0, aux=None):
        del key, disp
        alpha = jnp.asarray(0.5) if aux is None else jnp.asarray(aux)
        return jnp.asarray(eta) + alpha

    def update_nuisance(self, X, y, eta, disp, step_size=1.0, aux=None):
        del X, y, eta, disp, step_size
        return jnp.asarray(1.0), aux

    def init_nuisance(self):
        return jnp.asarray(1.0), jnp.asarray(0.5)


def test_fit_passes_grammar_nouns_to_custom_fitter() -> None:
    seen: dict[str, object] = {}
    expected = _make_fit_result()
    current_fitted_glm_type = importlib.import_module("glmax._fit").FittedGLM

    class RecordingFitter(AbstractFitter, strict=True):
        solver: AbstractLinearSolver = CholeskySolver()

        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            seen["model"] = model
            seen["data"] = data
            seen["init"] = init
            return expected

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))
    init = Params(beta=jnp.zeros(1), disp=jnp.array(0.0), aux=None)

    result = glmax.fit(model, data, init=init, fitter=RecordingFitter())

    assert isinstance(result, current_fitted_glm_type)
    assert bool(eqx.tree_equal(result.result, expected))
    assert isinstance(seen["model"], glmax.GLM)
    assert isinstance(seen["data"], GLMData)
    assert isinstance(seen["init"], Params)
    assert seen["init"]._fields == ("beta", "disp", "aux")


def test_fit_rejects_non_fitresult_from_custom_fitter() -> None:
    class BadFitter(AbstractFitter, strict=True):
        solver: AbstractLinearSolver = CholeskySolver()

        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> object:
            del model, data, init
            return object()

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((2, 1)), y=jnp.ones(2))

    with pytest.raises(TypeError, match="FitResult"):
        glmax.fit(model, data, fitter=BadFitter())


_DEFAULT_X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
# Gamma uses InverseLink (eta = 1/mu); X must not include zero to avoid eta=0 at init.
_GAMMA_X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
_INTERCEPT_ONLY_X = jnp.ones((5, 1))
_BINOMIAL_INTERCEPT_ONLY_X = jnp.ones((6, 1))
_VALID_LINK_COMBINATION_CASES = [
    pytest.param(
        Gaussian(IdentityLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([0.8, 1.0, 1.1, 1.3, 1.6]),
        id="gaussian-identity-link",
    ),
    pytest.param(
        Gaussian(LogLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([0.8, 1.0, 1.1, 1.3, 1.6]),
        id="gaussian-log-link",
    ),
    pytest.param(
        Gaussian(PowerLink(0.5)),
        _INTERCEPT_ONLY_X,
        jnp.array([0.8, 1.0, 1.1, 1.3, 1.6]),
        id="gaussian-power-link",
    ),
    pytest.param(
        Poisson(IdentityLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([1.0, 2.0, 3.0, 2.0, 1.0]),
        id="poisson-identity-link",
    ),
    pytest.param(
        Poisson(LogLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([1.0, 2.0, 3.0, 2.0, 1.0]),
        id="poisson-log-link",
    ),
    pytest.param(
        Binomial(LogitLink()),
        _BINOMIAL_INTERCEPT_ONLY_X,
        jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        id="binomial-logit-link",
    ),
    pytest.param(
        Binomial(IdentityLink()),
        _BINOMIAL_INTERCEPT_ONLY_X,
        jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        id="binomial-identity-link",
    ),
    pytest.param(
        Binomial(LogLink()),
        _BINOMIAL_INTERCEPT_ONLY_X,
        jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        id="binomial-log-link",
    ),
    pytest.param(
        NegativeBinomial(IdentityLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([1.0, 2.0, 4.0, 3.0, 2.0]),
        id="negative-binomial-identity-link",
    ),
    pytest.param(
        NegativeBinomial(LogLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([1.0, 2.0, 4.0, 3.0, 2.0]),
        id="negative-binomial-log-link",
    ),
    pytest.param(
        NegativeBinomial(NBLink(alpha=0.1)),
        _INTERCEPT_ONLY_X,
        jnp.array([1.0, 2.0, 4.0, 3.0, 2.0]),
        id="negative-binomial-nb-link",
    ),
    pytest.param(
        NegativeBinomial(PowerLink(0.5)),
        _INTERCEPT_ONLY_X,
        jnp.array([1.0, 2.0, 4.0, 3.0, 2.0]),
        id="negative-binomial-power-link",
    ),
    pytest.param(
        Gamma(IdentityLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([0.8, 1.0, 1.5, 2.0, 2.4]),
        id="gamma-identity-link",
    ),
    pytest.param(
        Gamma(InverseLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([0.8, 1.0, 1.5, 2.0, 2.4]),
        id="gamma-inverse-link",
    ),
    pytest.param(
        Gamma(LogLink()),
        _INTERCEPT_ONLY_X,
        jnp.array([0.8, 1.0, 1.5, 2.0, 2.4]),
        id="gamma-log-link",
    ),
]


def _assert_canonical_params_for_family(family, params: Params) -> None:
    assert params._fields == ("beta", "disp", "aux")
    assert params.beta.ndim == 1
    assert jnp.asarray(params.disp).shape == ()

    if isinstance(family, NegativeBinomial):
        assert jnp.allclose(params.disp, jnp.array(1.0))
        assert params.aux is not None
        assert float(jnp.asarray(params.aux)) > 0.0
        return

    assert params.aux is None
    if isinstance(family, Gaussian):
        assert float(jnp.asarray(params.disp)) > 0.0
    else:
        assert jnp.allclose(params.disp, jnp.array(1.0))


def _statsmodels_result_for_fitted(fitted: FittedGLM):
    family = fitted.model.family
    glink = family.glink
    scale = float(np.asarray(fitted.params.disp))
    aux = None if fitted.params.aux is None else float(np.asarray(fitted.params.aux))

    if isinstance(family, Gaussian):
        if isinstance(glink, IdentityLink):
            sm_family = sm.families.Gaussian(link=sm_links.Identity())
        elif isinstance(glink, LogLink):
            sm_family = sm.families.Gaussian(link=sm_links.Log())
        else:
            sm_family = sm.families.Gaussian(
                link=sm_links.Power(power=float(np.asarray(glink.power))),
                check_link=False,
            )
    elif isinstance(family, Poisson):
        if isinstance(glink, IdentityLink):
            sm_family = sm.families.Poisson(link=sm_links.Identity())
        else:
            sm_family = sm.families.Poisson(link=sm_links.Log())
    elif isinstance(family, Binomial):
        if isinstance(glink, LogitLink):
            sm_family = sm.families.Binomial(link=sm_links.Logit())
        elif isinstance(glink, LogLink):
            sm_family = sm.families.Binomial(link=sm_links.Log())
        else:
            sm_family = sm.families.Binomial(link=sm_links.Identity())
    elif isinstance(family, Gamma):
        if isinstance(glink, InverseLink):
            sm_family = sm.families.Gamma(link=sm_links.InversePower())
        elif isinstance(glink, LogLink):
            sm_family = sm.families.Gamma(link=sm_links.Log())
        else:
            sm_family = sm.families.Gamma(link=sm_links.Identity())
    elif isinstance(family, NegativeBinomial):
        if isinstance(glink, NBLink):
            sm_link = sm_links.NegativeBinomial(alpha=float(np.asarray(glink.alpha)))
        elif isinstance(glink, LogLink):
            sm_link = sm_links.Log()
        elif isinstance(glink, IdentityLink):
            sm_link = sm_links.Identity()
        else:
            sm_link = sm_links.Power(power=float(np.asarray(glink.power)))
        sm_family = sm.families.NegativeBinomial(alpha=aux, link=sm_link)
    else:
        raise TypeError(f"Unsupported statsmodels reference family: {type(family).__name__}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DomainWarning)
        return sm.GLM(np.asarray(fitted.y), np.asarray(fitted.X), family=sm_family).fit(scale=scale)


def _statsmodels_expected_pvalues(fitted: FittedGLM, sm_result) -> np.ndarray:
    if isinstance(fitted.model.family, Gaussian):
        df = fitted.X.shape[0] - fitted.params.beta.shape[0]
        return np.asarray(2.0 * t_cdf(-jnp.abs(jnp.asarray(sm_result.tvalues)), df))
    return np.asarray(sm_result.pvalues)


@pytest.mark.parametrize(
    ("family", "X", "y"),
    [
        (Gaussian(), _DEFAULT_X, jnp.array([0.2, 1.1, 2.0, 3.3])),
        (Poisson(), _DEFAULT_X, jnp.array([0.0, 1.0, 1.0, 2.0])),
        (Binomial(), _DEFAULT_X, jnp.array([0.0, 1.0, 0.0, 1.0])),
        (NegativeBinomial(), _DEFAULT_X, jnp.array([0.0, 1.0, 2.0, 4.0])),
        (Gamma(), _GAMMA_X, jr.gamma(jr.PRNGKey(1), 2.0, shape=(4,))),
    ],
)
def test_default_fitter_returns_canonical_fitresult_for_supported_families(family, X, y) -> None:
    current_fitted_glm_type = importlib.import_module("glmax._fit").FittedGLM
    model = glmax.specify(family=family)
    data = GLMData(X=X, y=y)

    result = glmax.fit(model, data)

    assert isinstance(result, current_fitted_glm_type)
    assert result.params.beta.shape == (1,)
    assert result.eta.shape == y.shape
    assert result.mu.shape == y.shape
    assert result.glm_wt.shape == y.shape
    assert result.score_residual.shape == y.shape
    _assert_canonical_params_for_family(family, result.params)


@pytest.mark.parametrize(("family", "X", "y"), _VALID_LINK_COMBINATION_CASES)
def test_fit_and_infer_succeed_across_all_supported_family_link_combinations(family, X, y) -> None:
    model = glmax.specify(family=family)
    data = GLMData(X=X, y=y)

    fitted = glmax.fit(model, data)
    inferred = glmax.infer(fitted)

    assert isinstance(fitted, FittedGLM)
    assert bool(fitted.converged)
    assert fitted.params.beta.shape == (1,)
    assert fitted.eta.shape == y.shape
    assert fitted.mu.shape == y.shape
    assert fitted.glm_wt.shape == y.shape
    assert fitted.score_residual.shape == y.shape
    assert jnp.all(jnp.isfinite(fitted.params.beta))
    assert bool(jnp.isfinite(fitted.params.disp))
    assert jnp.all(jnp.isfinite(fitted.eta))
    assert jnp.all(jnp.isfinite(fitted.mu))
    assert jnp.all(jnp.isfinite(fitted.glm_wt))
    assert jnp.all(jnp.isfinite(fitted.score_residual))
    assert bool(jnp.isfinite(fitted.objective))
    assert bool(jnp.isfinite(fitted.objective_delta))
    _assert_canonical_params_for_family(family, fitted.params)

    if isinstance(family, NegativeBinomial):
        assert fitted.params.aux is not None
        assert bool(jnp.isfinite(fitted.params.aux))
    else:
        assert fitted.params.aux is None

    assert isinstance(inferred, InferenceResult)
    assert bool(eqx.tree_equal(inferred.params, fitted.params))
    _assert_canonical_params_for_family(family, inferred.params)
    assert inferred.se.shape == fitted.params.beta.shape
    assert inferred.stat.shape == fitted.params.beta.shape
    assert inferred.p.shape == fitted.params.beta.shape
    assert jnp.all(jnp.isfinite(inferred.se))
    assert jnp.all(jnp.isfinite(inferred.stat))
    assert jnp.all(jnp.isfinite(inferred.p))
    assert jnp.all((inferred.p >= 0.0) & (inferred.p <= 1.0))


@pytest.mark.parametrize(("family", "X", "y"), _VALID_LINK_COMBINATION_CASES)
def test_fit_and_infer_match_statsmodels_across_all_supported_family_link_combinations(family, X, y) -> None:
    fitted = glmax.fit(glmax.specify(family=family), GLMData(X=X, y=y))
    inferred = glmax.infer(fitted)
    sm_result = _statsmodels_result_for_fitted(fitted)
    sm_p = _statsmodels_expected_pvalues(fitted, sm_result)

    assert_allclose(np.asarray(fitted.params.beta), np.asarray(sm_result.params), rtol=1e-4, atol=1e-5)
    assert_allclose(np.asarray(inferred.se), np.asarray(sm_result.bse), rtol=1e-4, atol=1e-5)
    assert_allclose(np.asarray(inferred.stat), np.asarray(sm_result.tvalues), rtol=1e-4, atol=1e-5)
    assert_allclose(np.asarray(inferred.p), sm_p, rtol=1e-4, atol=1e-5)


def test_fit_boundary_rejects_raw_data_and_non_params_init() -> None:
    with pytest.raises(TypeError, match="GLM"):
        glmax.fit(object(), GLMData(X=jnp.ones((3, 1)), y=jnp.ones(3)))

    with pytest.raises(TypeError, match="GLMData"):
        glmax.fit(glmax.GLM(), jnp.ones((3, 1)))

    with pytest.raises(TypeError, match="Params"):
        glmax.fit(glmax.GLM(), GLMData(X=jnp.ones((3, 1)), y=jnp.ones(3)), init=jnp.zeros(1))


@pytest.mark.parametrize(
    ("bad_init", "error_type", "match"),
    [
        (
            Params(beta=["bad"], disp=jnp.array(0.0), aux=None),
            TypeError,
            "Params.beta must be numeric",
        ),
        (
            Params(beta=jnp.array([1.0]), disp="bad", aux=None),
            TypeError,
            "Params.disp must be numeric",
        ),
        (
            Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux="bad"),
            TypeError,
            "Params.aux must be numeric",
        ),
        (
            Params(beta=jnp.array([1.0]), disp=jnp.array(0, dtype=jnp.int32), aux=None),
            TypeError,
            "Params.disp must have an inexact dtype",
        ),
        (
            Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux=jnp.array(0, dtype=jnp.int32)),
            TypeError,
            "Params.aux must have an inexact dtype",
        ),
        (
            Params(beta=jnp.array([1.0, 2.0]), disp=jnp.array(0.0), aux=None),
            ValueError,
            "Params.beta must be a one-dimensional vector with length equal to X.shape\\[1\\]",
        ),
        (
            Params(beta=jnp.array([1.0]), disp=jnp.array([0.0, 1.0]), aux=None),
            ValueError,
            "Params.disp must be a scalar",
        ),
        (
            Params(beta=jnp.array([1.0]), disp=jnp.array(0.0), aux=jnp.array([0.0, 1.0])),
            ValueError,
            "Params.aux must be a scalar",
        ),
    ],
)
def test_fit_validates_init_params_at_public_boundary_before_custom_fitter(
    bad_init: Params,
    error_type: type[Exception],
    match: str,
) -> None:
    seen = {"called": False}

    class RecordingFitter(AbstractFitter, strict=True):
        solver: AbstractLinearSolver = CholeskySolver()

        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            del model, data
            seen["called"] = True
            seen["init"] = init
            return _make_fit_result()

    model = glmax.GLM()
    data = GLMData(X=jnp.ones((3, 1)), y=jnp.ones(3))

    with pytest.raises(error_type, match=match):
        glmax.fit(model, data, init=bad_init, fitter=RecordingFitter())

    assert not seen["called"]


@pytest.mark.parametrize("family", [Gaussian(), Gamma(), Poisson(), Binomial()])
def test_fit_ignores_aux_for_families_without_aux_state_before_custom_fitter(family) -> None:
    seen = {"called": False}

    class RecordingFitter(AbstractFitter, strict=True):
        solver: AbstractLinearSolver = CholeskySolver()

        def __call__(self, model: glmax.GLM, data: GLMData, init: Params | None = None) -> FitResult:
            del model, data
            seen["called"] = True
            assert init is not None
            return FitResult(
                params=init,
                X=jnp.array([[1.0], [1.0]]),
                y=jnp.array([1.0, 1.0]),
                eta=jnp.array([1.0, 1.0]),
                mu=jnp.array([1.0, 1.0]),
                glm_wt=jnp.array([1.0, 1.0]),
                converged=jnp.array(True),
                num_iters=jnp.array(1),
                objective=jnp.array(0.0),
                objective_delta=jnp.array(-1e-4),
                score_residual=jnp.array([0.0, 0.0]),
            )

    model = glmax.GLM(family=family)
    data = GLMData(X=jnp.ones((3, 1)), y=jnp.ones(3))
    init = Params(beta=jnp.zeros(1), disp=jnp.array(1.0), aux=jnp.array(0.2))

    result = glmax.fit(model, data, init=init, fitter=RecordingFitter())

    assert isinstance(result, FittedGLM)
    assert seen["called"]
    assert result.params.aux is None
    assert jnp.allclose(result.params.disp, jnp.array(1.0))


def test_fitter_is_abstract_equinox_model() -> None:
    assert issubclass(AbstractFitter, eqx.Module)

    with pytest.raises(TypeError):
        AbstractFitter()


def test_canonical_fit_supports_non_default_solver_path() -> None:
    model = glmax.specify(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0, 0.5], [1.0, 1.5], [1.0, 2.0], [1.0, 3.0]]),
        y=jnp.array([0.8, 1.7, 2.1, 2.9]),
    )

    result = glmax.fit(model, data, fitter=IRLSFitter(solver=QRSolver()))

    assert isinstance(result, FittedGLM)
    assert result.params.beta.shape == (2,)
    assert bool(result.converged)
    assert jnp.all(jnp.isfinite(result.params.beta))


def test_default_fitter_forwards_offset_and_transforms_init_to_eta() -> None:
    model = glmax.GLM(family=Gaussian())
    X = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.5, -1.0]])
    y = jnp.array([1.0, 0.0, 1.0])
    offset = jnp.array([0.2, 0.1, 0.3])
    init = Params(beta=jnp.array([0.4, -0.1]), disp=jnp.array(0.7), aux=None)

    data = GLMData(X=X, y=y, offset=offset)
    result_1 = glmax.fit(model, data, init=init)
    result_2 = glmax.fit(model, data, init=init)

    assert isinstance(result_1, FittedGLM)
    assert jnp.allclose(result_1.beta, result_2.beta)
    assert jnp.allclose(result_1.params.disp, result_2.params.disp)
    assert result_1.params.aux is None
    assert result_2.params.aux is None


def test_irls_fitter_canonicalizes_supported_warm_start_params() -> None:
    model = glmax.GLM(family=_CanonicalWarmStartFamily())
    data = GLMData(X=jnp.array([[1.0], [2.0], [3.0], [4.0]]), y=jnp.array([1.2, 1.9, 3.1, 4.0]))
    raw_init = Params(beta=jnp.array([0.1]), disp=jnp.array(0.7), aux=jnp.array(0.1))
    canonical_init = Params(beta=jnp.array([0.1]), disp=jnp.array(1.0), aux=jnp.array(0.25))

    raw_result = IRLSFitter()(model, data, init=raw_init)
    canonical_result = IRLSFitter()(model, data, init=canonical_init)

    assert jnp.allclose(raw_result.params.disp, canonical_result.params.disp)
    assert jnp.allclose(raw_result.params.aux, canonical_result.params.aux)
    assert jnp.allclose(raw_result.objective, canonical_result.objective)


def test_public_fit_matches_single_canonicalization_reference_for_non_idempotent_warm_starts() -> None:
    model = glmax.GLM(family=_NonIdempotentCanonicalWarmStartFamily())
    data = GLMData(X=jnp.array([[1.0], [2.0], [3.0], [4.0]]), y=jnp.array([1.2, 1.9, 3.1, 4.0]))
    seed = Params(beta=jnp.array([0.1]), disp=jnp.array(0.7), aux=jnp.array(2.0))

    expected = IRLSFitter()(model, data, init=seed)
    first = glmax.fit(model, data, init=seed)
    inferred = glmax.infer(first)
    second = glmax.fit(model, data, init=first.params)

    assert jnp.allclose(first.glm_wt, expected.glm_wt)
    assert jnp.allclose(first.objective, expected.objective)
    assert jnp.allclose(first.params.disp, expected.params.disp)
    assert jnp.allclose(first.params.aux, expected.params.aux)
    assert jnp.allclose(second.params.disp, first.params.disp)
    assert jnp.allclose(second.params.aux, first.params.aux)
    assert jnp.allclose(inferred.params.disp, first.params.disp)
    assert jnp.allclose(inferred.params.aux, first.params.aux)


def test_default_fitter_initializes_aux_for_aux_aware_families() -> None:
    model = glmax.GLM(family=_AuxSensitiveIRLSFamily())
    data = GLMData(X=jnp.array([[1.0], [1.0], [1.0]]), y=jnp.array([1.0, 2.0, 3.0]))

    result = glmax.fit(model, data)

    assert isinstance(result, FittedGLM)
    assert jnp.allclose(result.params.disp, jnp.array(1.0))
    assert result.params.aux is not None
    assert jnp.allclose(result.params.aux, jnp.array(0.5))


def test_irls_fitter_threads_aux_through_kernel_state() -> None:
    model = glmax.GLM(family=_AuxSensitiveIRLSFamily())
    data = GLMData(X=jnp.array([[1.0], [1.0], [1.0]]), y=jnp.array([1.0, 2.0, 3.0]))
    small_aux = Params(beta=jnp.array([0.0]), disp=jnp.array(1.0), aux=jnp.array(0.5))
    large_aux = Params(beta=jnp.array([0.0]), disp=jnp.array(1.0), aux=jnp.array(2.0))

    small_fit = IRLSFitter()(model, data, init=small_aux)
    large_fit = IRLSFitter()(model, data, init=large_aux)

    assert jnp.allclose(small_fit.params.aux, small_aux.aux)
    assert jnp.allclose(large_fit.params.aux, large_aux.aux)
    assert not jnp.allclose(small_fit.glm_wt, large_fit.glm_wt)
    assert not jnp.allclose(small_fit.objective, large_fit.objective)


def test_negative_binomial_fit_glm_weights_match_final_auxiliary_parameter() -> None:
    model = glmax.GLM(family=NegativeBinomial())
    data = GLMData(X=_DEFAULT_X, y=jnp.array([0.0, 1.0, 2.0, 4.0]))

    fitted = glmax.fit(model, data)
    _, _, expected_weight = model.working_weights(fitted.eta, fitted.params.disp, fitted.params.aux)

    assert fitted.params.aux is not None
    assert jnp.allclose(fitted.glm_wt, expected_weight, rtol=1e-6, atol=1e-7)


def test_default_fitter_validates_init_beta_shape() -> None:
    X = jnp.ones((4, 2))
    y = jnp.ones(4)
    bad_init = Params(beta=jnp.ones((2, 1)), disp=jnp.array(0.0), aux=None)

    with pytest.raises(ValueError, match="Params.beta"):
        glmax.fit(glmax.GLM(), GLMData(X=X, y=y), init=bad_init)


@pytest.mark.parametrize(
    ("family", "y"),
    [
        (Gaussian(), jnp.array([0.1, 1.0, 2.1, 2.9, 4.2])),
        (Poisson(), jnp.array([0.0, 1.0, 1.0, 2.0, 3.0])),
        (Binomial(), jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])),
        (NegativeBinomial(), jnp.array([0.0, 1.0, 2.0, 1.0, 4.0])),
    ],
)
def test_all_families_succeed_with_default_fitter(family, y) -> None:
    data = GLMData(X=jnp.array([[0.0], [1.0], [2.0], [3.0], [4.0]]), y=y)
    result = glmax.fit(glmax.GLM(family=family), data)

    assert isinstance(result, FittedGLM)
    assert isinstance(result.params, Params)
    assert result.score_residual.shape == (data.n_samples,)
    assert bool(jnp.isfinite(result.objective))
    assert bool(jnp.isfinite(result.objective_delta))
    _assert_canonical_params_for_family(family, result.params)


def test_single_feature_beta_shape_roundtrip() -> None:
    model = glmax.GLM(family=Gaussian())
    X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    y = jnp.array([1.2, 1.9, 3.1, 4.0])
    data = GLMData(X=X, y=y)

    first = glmax.fit(model, data)
    assert first.beta.shape == (1,)

    second = glmax.fit(model, data, init=first.params)
    assert second.beta.shape == (1,)
    assert jnp.allclose(second.params.disp, first.params.disp)
    assert second.params.aux is None


def test_unsupported_weights_rejected() -> None:
    X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    y = jnp.array([0.2, 0.9, 2.2, 2.8])

    with pytest.raises(ValueError, match="weights"):
        glmax.fit(glmax.GLM(), GLMData(X=X, y=y, weights=jnp.ones(4)))


def test_specify_returns_glm_instance() -> None:
    model = specify()
    assert isinstance(model, glmax.GLM)
