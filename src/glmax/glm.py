# pattern: Functional Core

from typing import NamedTuple, Tuple

import equinox as eqx

from jax import Array, numpy as jnp
from jaxtyping import ArrayLike, ScalarLike

from .family.dist import ExponentialFamily, Gaussian, NegativeBinomial, Poisson
from .infer.contracts import AbstractLinearSolver
from .infer.fitters import AbstractGLMFitter, IRLSFitter
from .infer.inference import (
    AbstractStdErrEstimator,
    FisherInfoError,
    wald_test as inference_wald_test,
)
from .infer.solvers import CholeskySolver


class GLMState(NamedTuple):
    """
    Represents the state of a Generalized Linear Model (GLM) during fitting.
    This class stores the key parameters and intermediate results from
    a GLM estimation process.

    **Attributes:**

    - `beta`: Estimated regression coefficients.
    - `se`: Standard errors of the estimated coefficients.
    - `z`: Z-scores for hypothesis testing of each coefficient, computed as `beta / se`.
    - `p`: P-values associated with each coefficient.
    - `eta`: the transformed mean response, linear component eta.
    - `mu`: The **fitted mean response**, derived from the inverse link function applied to - `eta`.
    - `glm_wt`: weights used in the iterative weighted least squares procedure The weights used in the iterative
        weighted least squares (IWLS) procedure during GLM fitting.
    - `num_iters`: number of iterations taken for the optimization algorithm to converge.
    - `converged`: boolean indicating whether the optimization converged.
    - `infor_inv`: **inverse of the Fisher Information matrix**, used for score tests  # for score test
    - `resid`: The **residuals** from the model, used in score tests.
         ⚠ **Note** These are not the working residuals from the IWLS algorithm
    - `alpha`: The **dispersion parameter** in the Negative Binomial (NB) model, controlling overdispersion
    """

    beta: Array
    se: Array
    z: Array
    p: Array
    eta: Array
    mu: Array
    glm_wt: Array
    num_iters: Array
    converged: Array
    infor_inv: Array  # for score test
    resid: Array  # for score test, not the working resid!
    alpha: Array  # dispersion parameter in NB model


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
    fitter: AbstractGLMFitter = IRLSFitter()

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
        n, _ = X.shape
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
        return inference_wald_test(statistic, df, self.family)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike = 0.0,
        init: ArrayLike = None,
        alpha_init: ScalarLike = None,
        se_estimator: AbstractStdErrEstimator = FisherInfoError(),
        fitter: AbstractGLMFitter | None = None,
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
        from .fit import fit as module_fit

        return module_fit(
            X,
            y,
            family=self.family,
            solver=self.solver,
            fitter=self.fitter if fitter is None else fitter,
            offset_eta=offset_eta,
            init=init,
            alpha_init=alpha_init,
            se_estimator=se_estimator,
            max_iter=max_iter,
            tol=tol,
            step_size=step_size,
        )


GLM.__init__.__doc__ = r"""**Arguments:**

- `family`: An instance of [`ExponentialFamily`][] indicating the distribution of the response variable
    (e.g., Gaussian, Poisson, Negative Binomial). This determines the link function and variance structure.
- `solver`: An instance of [`AbstractLinearSolver`][] to use for solving the weighted least squares problem
    for inference.
"""
