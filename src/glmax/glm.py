# pattern: Functional Core

from __future__ import annotations

import inspect

from typing import Tuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike, ScalarLike

from .contracts import Diagnostics, FitResult, GLMData, Params
from .family.dist import ExponentialFamily, Gaussian, NegativeBinomial, Poisson
from .family.utils import t_cdf
from .infer.optimize import irls
from .infer.solve import AbstractLinearSolver, CholeskySolver
from .infer.stderr import AbstractStdErrEstimator, FisherInfoError


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

    def fit(
        self,
        data: GLMData,
        *legacy_args: ArrayLike,
        init_eta: ArrayLike = None,
        disp_init: ScalarLike = None,
        se_estimator: AbstractStdErrEstimator = FisherInfoError(),
        max_iter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 1.0,
        **legacy_kwargs: ArrayLike,
    ) -> FitResult:
        """
        Represents the fitted state of a Generalized Linear Model (GLM).

        This class stores the estimated parameters, standard errors, diagnostics,
        and other relevant information from the fitting process of a GLM.

        **Arguments:**

        - `data`: canonical covariate/response noun.
        - `init_eta`: optional initial linear predictor.
        - `disp_init`: optional canonical dispersion initial value.
        - `max_iter`: maximum number of iterations, default to 1000.
        - `tol`: tolerance for convergence, default to 1e-3.
        - `step_size`: step size, default to 1.0.

        **Returns:**

        -  A [`glmax.FitResult`][] containing the final estimated parameters and convergence diagnostics
            from the fitted GLM model.
        """
        if legacy_args:
            raise TypeError(
                "GLM.fit(...) no longer accepts separate `X, y` positional inputs; "
                "pass a single `GLMData(X=..., y=...)` value as `data`."
            )
        if "init" in legacy_kwargs:
            raise TypeError(
                "GLM.fit(...) no longer accepts legacy keyword `init`; "
                "pass `init_eta=` and optional `disp_init=` instead."
            )
        if "alpha_init" in legacy_kwargs:
            raise TypeError(
                "GLM.fit(...) no longer accepts legacy keyword `alpha_init`; " "use canonical `disp_init=` instead."
            )
        if legacy_kwargs:
            unexpected = ", ".join(f"`{name}`" for name in sorted(legacy_kwargs))
            raise TypeError(f"GLM.fit(...) got unexpected keyword argument(s): {unexpected}.")
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
            inferred_eta, inferred_disp = self.calc_eta_and_dispersion(effective_data, max_iter=max_iter)
            if init_eta is None:
                init_eta = inferred_eta
            if disp_init is None:
                disp_init = inferred_disp

        beta, n_iter, converged, disp, objective, objective_delta = irls(
            X_array,
            y_array,
            self.family,
            self.solver,
            init_eta,
            max_iter,
            tol,
            step_size,
            offset_array,
            disp_init=disp_init,
        )

        eta = X_array @ beta + offset_array
        mu = self.family.glink.inverse(eta)
        score_residual = (y_array - mu) * self.family.glink.deriv(mu)  # note: this is the working residual

        _, _, weight = self.family.calc_weight(X_array, y_array, eta, disp)

        curvature = se_estimator(self.family, X_array, y_array, eta, mu, weight, disp)
        beta_se = jnp.sqrt(jnp.diag(curvature))

        df = X_array.shape[0] - X_array.shape[1]
        beta = jnp.ravel(beta)  # (p,)
        stat = beta / beta_se

        pval_wald = self.wald_test(stat, df)

        return FitResult(
            params=Params(beta=beta, disp=self.family.canonical_dispersion(disp)),
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


GLM.fit.__signature__ = inspect.Signature(
    parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("data", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=GLMData),
        inspect.Parameter("init_eta", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=ArrayLike),
        inspect.Parameter("disp_init", inspect.Parameter.KEYWORD_ONLY, default=None, annotation=ScalarLike),
        inspect.Parameter(
            "se_estimator",
            inspect.Parameter.KEYWORD_ONLY,
            default=FisherInfoError(),
            annotation=AbstractStdErrEstimator,
        ),
        inspect.Parameter("max_iter", inspect.Parameter.KEYWORD_ONLY, default=1000, annotation=int),
        inspect.Parameter("tol", inspect.Parameter.KEYWORD_ONLY, default=1e-3, annotation=float),
        inspect.Parameter("step_size", inspect.Parameter.KEYWORD_ONLY, default=1.0, annotation=float),
    ],
    return_annotation=FitResult,
)

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
