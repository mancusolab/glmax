# pattern: Functional Core

from __future__ import annotations

import inspect

from typing import Tuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike, ScalarLike

from .data import GLMData
from .family.dist import ExponentialFamily, Gaussian, NegativeBinomial, Poisson
from .family.utils import t_cdf
from .fit import FitResult, Params
from .infer.diagnostics import Diagnostics
from .infer.optimize import irls
from .infer.solve import AbstractLinearSolver, CholeskySolver
from .infer.stderr import AbstractStdErrEstimator, FisherInfoError


_REMOVED_KEYWORD = object()


def _bound_fit_signature() -> inspect.Signature:
    signature = inspect.signature(_fit_impl)
    parameters = list(signature.parameters.values())[1:]
    return signature.replace(parameters=parameters)


def _validate_bound_fit_args(args: tuple[object, ...]) -> None:
    if len(args) <= 1:
        return
    extra_arg = args[1]
    if isinstance(extra_arg, Params):
        raise TypeError(
            "GLM.fit(...) no longer accepts positional `Params`. "
            "Use `glmax.fit(model, data, init=params)` for Params-based initialization."
        )
    raise TypeError(
        "GLM.fit(...) accepts exactly one positional argument after binding: `data`. "
        "Use `init_eta=` for a linear predictor or `disp_init=` for dispersion initialization."
    )


def _fit_model(
    model: "GLM",
    data: GLMData,
    *,
    init_eta: ArrayLike = None,
    disp_init: ScalarLike = None,
    se_estimator: AbstractStdErrEstimator = FisherInfoError(),
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
) -> FitResult:
    """Canonical GLM fit kernel used by the public grammar verb."""
    if not isinstance(data, GLMData):
        raise TypeError("GLM.fit(...) expects `data` to be a GLMData instance.")
    if data.weights is not None:
        raise ValueError("GLMData.weights is not supported in GLM.fit yet.")
    X_array, y_array, offset_array, _, _ = data.canonical_arrays()
    if X_array.shape[0] == 0:
        raise ValueError("GLMData.mask removes all samples; at least one effective sample is required.")
    effective_data = GLMData(X=X_array, y=y_array, offset=offset_array)

    if init_eta is not None:
        init_eta = jnp.asarray(init_eta)
        if init_eta.ndim != 1 or init_eta.shape[0] != X_array.shape[0]:
            raise ValueError("init_eta must be a one-dimensional vector with length equal to sample count.")
        if not bool(jnp.all(jnp.isfinite(init_eta))):
            raise ValueError("init_eta must contain only finite values.")

    if disp_init is not None:
        disp_init = jnp.asarray(disp_init)
        if disp_init.ndim > 0 and disp_init.size != 1:
            raise ValueError("disp_init must be a scalar dispersion value.")
        if not bool(jnp.all(jnp.isfinite(disp_init))):
            raise ValueError("disp_init must contain only finite values.")

    if init_eta is None or disp_init is None:
        inferred_eta, inferred_disp = model.calc_eta_and_dispersion(effective_data, max_iter=max_iter)
        if init_eta is None:
            init_eta = inferred_eta
        if disp_init is None:
            disp_init = inferred_disp

    beta, n_iter, converged, disp, objective, objective_delta = irls(
        X_array,
        y_array,
        model.family,
        model.solver,
        init_eta,
        max_iter,
        tol,
        step_size,
        offset_array,
        disp_init=disp_init,
    )

    eta = X_array @ beta + offset_array
    mu = model.family.glink.inverse(eta)
    score_residual = (y_array - mu) * model.family.glink.deriv(mu)

    _, _, weight = model.family.calc_weight(eta, disp)
    curvature = se_estimator(model.family, X_array, y_array, eta, mu, weight, disp)
    beta_se = jnp.sqrt(jnp.diag(curvature))

    df = X_array.shape[0] - X_array.shape[1]
    beta = jnp.ravel(beta)
    stat = beta / beta_se
    pval_wald = model.wald_test(stat, df)

    return FitResult(
        params=Params(beta=beta, disp=model.family.canonical_dispersion(disp)),
        se=beta_se,
        z=stat,
        p=pval_wald,
        eta=eta,
        mu=mu,
        glm_wt=weight,
        diagnostics=Diagnostics(
            converged=converged,
            num_iters=n_iter,
            objective=objective,
            objective_delta=objective_delta,
        ),
        curvature=curvature,
        score_residual=score_residual,
    )


def _fit_impl(
    model: "GLM",
    data: GLMData,
    init_eta: ArrayLike = None,
    disp_init: ScalarLike = None,
    se_estimator: AbstractStdErrEstimator = FisherInfoError(),
    max_iter: int = 1000,
    tol: float = 1e-3,
    step_size: float = 1.0,
) -> FitResult:
    """Internal convenience method over the canonical GLM fit kernel.

    Use `glmax.fit(model, data, init=...)` for the public grammar contract.

    **Arguments:**

    - `data`: canonical covariate/response noun.
    - `init_eta`: optional initial linear predictor.
    - `disp_init`: optional canonical dispersion initial value.
    - `max_iter`: maximum number of iterations, default to 1000.
    - `tol`: tolerance for convergence, default to 1e-3.
    - `step_size`: step size, default to 1.0.

    **Returns:**

    -  A [`glmax.FitResult`][] containing the fitted parameter and diagnostic artifacts.
    """
    return _fit_model(
        model,
        data,
        init_eta=init_eta,
        disp_init=disp_init,
        se_estimator=se_estimator,
        max_iter=max_iter,
        tol=tol,
        step_size=step_size,
    )


class _GLMBoundFit:
    def __init__(self, instance: "GLM") -> None:
        self._instance = instance
        self.__wrapped__ = _fit_impl
        self.__doc__ = _fit_impl.__doc__
        self.__signature__ = _bound_fit_signature()

    def __call__(self, *args: object, **kwargs: object) -> FitResult:
        _validate_bound_fit_args(args)
        if "init" in kwargs:
            raise TypeError(
                "GLM.fit(...) no longer accepts `init`. "
                "Use `glmax.fit(model, data, init=...)` for Params-based initialization, "
                "or pass `init_eta=` when providing a linear predictor directly."
            )
        if "alpha_init" in kwargs:
            raise TypeError("GLM.fit(...) no longer accepts `alpha_init`. Use `disp_init=` instead.")

        # Bind args/kwargs to the documented signature so defaults are applied.
        bound = _bound_fit_signature().bind(*args, **kwargs)
        bound.apply_defaults()
        ba = bound.arguments

        data = ba["data"]
        init_eta = ba.get("init_eta")
        disp_init = ba.get("disp_init")
        se_estimator = ba.get("se_estimator", FisherInfoError())
        max_iter = ba.get("max_iter", 1000)
        tol = ba.get("tol", 1e-3)
        step_size = ba.get("step_size", 1.0)

        from .fit import fit as _glmax_fit

        def _delegating_fitter(model: "GLM", data: GLMData, init: Params | None = None) -> FitResult:
            return _fit_model(
                model,
                data,
                init_eta=init_eta,
                disp_init=disp_init,
                se_estimator=se_estimator,
                max_iter=max_iter,
                tol=tol,
                step_size=step_size,
            )

        return _glmax_fit(self._instance, data, fitter=_delegating_fitter)


class _GLMFitDescriptor:
    __wrapped__ = _fit_impl
    __doc__ = _fit_impl.__doc__

    def __call__(self, model: "GLM", *args: object, **kwargs: object) -> FitResult:
        return _GLMBoundFit(model)(*args, **kwargs)

    def __get__(self, instance: "GLM" | None, owner: type["GLM"]) -> object:
        del owner
        if instance is None:
            return self
        return _GLMBoundFit(instance)


class GLM(eqx.Module):
    """
    This class provides a flexible framework for fitting Generalized Linear Models (GLMs),
    which extend linear regression to accommodate response variables from the
    Exponential Family (e.g., Gaussian, Poisson, Binomial). The GLM framework allows for
    different link functions and estimation methods.

    !!! info


    """

    family: ExponentialFamily = Gaussian()
    solver: AbstractLinearSolver = CholeskySolver()

    def calc_eta_and_dispersion(
        self,
        data: GLMData,
        max_iter: int = 1000,
    ) -> Tuple[Array, Array]:
        """Calculate eta and canonical dispersion value.

        :param data: canonical GLM data noun
        :return: eta and canonical dispersion value
        """
        X, y, offset_eta, _, _ = data.canonical_arrays()
        n, _ = X.shape
        init_val = self.family.init_eta(y)
        if isinstance(self.family, NegativeBinomial):
            jaxqtl_pois = GLM(family=Poisson())
            glm_state_pois = jaxqtl_pois.fit(
                GLMData(X=X, y=y, offset=offset_eta),
                init_eta=init_val,
                max_iter=max_iter,
            )

            # fit covariate-only model (null)
            alpha_init = n / jnp.sum((y / self.family.glink.inverse(glm_state_pois.eta) - 1) ** 2)
            eta = glm_state_pois.eta
            disp = self.family.estimate_dispersion(X, y, eta, alpha=1.0 / alpha_init, max_iter=max_iter)

            # convert disp to 0.1 if bad initialization
            disp = self.family.canonical_dispersion(jnp.nan_to_num(disp, nan=0.1))

        else:
            eta = init_val
            disp = self.family.canonical_dispersion(jnp.asarray(0.0))

        return eta, disp

    def wald_test(self, statistic: ArrayLike, df: int) -> Array:
        """
        Computes the Wald test statistic and corresponding p-value.

        The Wald test is used to assess the significance of estimated coefficients
        in a regression model. It tests the null hypothesis that a parameter (or
        set of parameters) is equal to zero.

        Under the assumption that the **Maximum Likelihood Estimator (MLE)** follows:

        `statistic: The test statistic, typically beta / SE(beta), where `SE` is
        the standard error of the estimated coefficient.

        `df: The degrees of freedom associated with the test.
        For a single coefficient, `df=1`, whereas for a joint test involving multiple coefficients,
        `df` corresponds to the number of parameters tested.

        Returns:
        :return: The Wald test statistic's corresponding p-value.
        """
        if isinstance(self.family, Gaussian):
            pval = 2 * t_cdf(-abs(statistic), df)
        else:
            pval = 2 * norm.sf(abs(statistic))

        return pval

    fit = _GLMFitDescriptor()


GLM.__init__.__doc__ = r"""**Arguments:**

- `family`: An instance of [`ExponentialFamily`][] indicating the distribution of the response variable
    (e.g., Gaussian, Poisson, Negative Binomial). This determines the link function and variance structure.
- `solver`: An instance of [`AbstractLinearSolver`][] to use for solving the weighted least squares problem
    for inference.
"""


def specify(*, family: ExponentialFamily | None = None, solver: AbstractLinearSolver | None = None) -> GLM:
    """Construct a GLM explicitly from grammar-style arguments."""
    kwargs = {}
    if family is not None:
        kwargs["family"] = family
    if solver is not None:
        kwargs["solver"] = solver
    return GLM(**kwargs)
