import os
import warnings

from typing import Tuple

import equinox as eqx

from jax import Array, numpy as jnp
from jaxtyping import ArrayLike, ScalarLike

from .family.dist import ExponentialFamily, Gaussian, NegativeBinomial, Poisson
from .infer.result import GLMState
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
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike = 0.0,
        max_iter: int = 1000,
    ) -> Tuple[Array, Array]:
        """Calculate eta and dispersion parameter alpha

        :param X: covariate data matrix (nxp)
        :param y: outcome vector (nx1)
        :param offset_eta: offset (nx1)
        :return: eta, dispersion (alpha)
        """
        n, p = X.shape
        init_val = self.family.init_eta(y)
        if isinstance(self.family, NegativeBinomial):
            jaxqtl_pois = GLM(family=Poisson())
            glm_state_pois = jaxqtl_pois.fit(X, y, init=init_val, offset_eta=offset_eta, max_iter=max_iter)

            # fit covariate-only model (null)
            alpha_init = n / jnp.sum((y / self.family.glink.inverse(glm_state_pois.eta) - 1) ** 2)
            eta = glm_state_pois.eta
            disp = self.family.estimate_dispersion(X, y, eta, alpha=1.0 / alpha_init, max_iter=max_iter)

            # convert disp to 0.1 if bad initialization
            disp = jnp.nan_to_num(disp, nan=0.1)

        else:
            eta = init_val
            disp = jnp.asarray(0.0)  # alpha is non-zero only in NB model

        return eta, disp

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike = 0.0,
        init: ArrayLike = None,
        alpha_init: ScalarLike = None,
        se_estimator: AbstractStdErrEstimator = FisherInfoError(),
        max_iter: int = 1000,
        tol: float = 1e-3,
        step_size: float = 1.0,
    ) -> GLMState:
        """
        Represents the fitted state of a Generalized Linear Model (GLM).

        This class stores the estimated parameters, standard errors, diagnostics,
        and other relevant information from the fitting process of a GLM.

        **Arguments:**

        - `X`: covariate data matrix.
        - `y`: outcome vector.
        - `init`: initial value for betas.
        - `max_iter`: maximum number of iterations, default to 1000.
        - `tol`: tolerance for convergence, default to 1e-3.
        - `step_size`: step size, default to 1.0.
        - `offset_eta`: offset.
        - `alpha_init`: initial value for alpha in NB model, default to 0s.

        **Returns:**

        -  A [`glmax.GLMState`][] containing the final estimated parameters and convergence diagnostics
            from the fitted GLM model.
        """
        if os.environ.get("GLMAX_WARN_GLM_FIT_COMPAT", "").lower() in {"1", "true", "yes"}:
            warnings.warn(
                "GLM.fit is a compatibility wrapper over glmax.fit; prefer glmax.fit for new code.",
                UserWarning,
                stacklevel=2,
            )

        from .fit import fit as gx_fit

        if jnp.ndim(offset_eta) == 0:
            offset = None if float(offset_eta) == 0.0 else jnp.full((X.shape[0],), offset_eta)
        else:
            offset = offset_eta

        return gx_fit(
            self,
            X,
            y,
            offset=offset,
            covariance=se_estimator,
            init=init,
            options={
                "alpha_init": alpha_init,
                "max_iter": max_iter,
                "tol": tol,
                "step_size": step_size,
            },
        )


GLM.__init__.__doc__ = r"""**Arguments:**

- `family`: An instance of [`ExponentialFamily`][] indicating the distribution of the response variable
    (e.g., Gaussian, Poisson, Negative Binomial). This determines the link function and variance structure.
- `solver`: An instance of [`AbstractLinearSolver`][] to use for solving the weighted least squares problem
    for inference.
"""
