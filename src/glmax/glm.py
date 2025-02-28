from typing import NamedTuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike, ScalarLike

from .family.dist import ExponentialFamily, Gaussian
from .family.utils import t_cdf
from .infer.optimize import irls
from .infer.solve import AbstractLinearSolver, CholeskySolver
from .infer.stderr import AbstractStdErrEstimator, FisherInfoError


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
        X: ArrayLike,
        y: ArrayLike,
        offset_eta: ArrayLike = 0.0,
        init: ArrayLike = None,
        alpha_init: ScalarLike = 0.0,
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
        beta, n_iter, converged, alpha = irls(
            X,
            y,
            self.family,
            self.solver,
            init,
            max_iter,
            tol,
            step_size,
            offset_eta,
            alpha_init,
        )

        eta = X @ beta + offset_eta
        mu = self.family.glink.inverse(eta)
        resid = (y - mu) * self.family.glink.deriv(mu)  # note: this is the working resid

        _, _, weight = self.family.calc_weight(X, y, eta, alpha)

        resid_covar = se_estimator(self.family, X, y, eta, mu, weight, alpha)
        beta_se = jnp.sqrt(jnp.diag(resid_covar))

        df = X.shape[0] - X.shape[1]
        beta = beta.squeeze()  # (p,)
        stat = beta / beta_se

        pval_wald = self.wald_test(stat, df)

        return GLMState(
            beta,
            beta_se,
            stat,
            pval_wald,
            eta,
            mu,
            weight,
            n_iter,
            converged,
            resid_covar,
            resid,
            alpha,
        )


GLM.__init__.__doc__ = r"""**Arguments:**

- `family`: An instance of [`ExponentialFamily`][] indicating the distribution of the response variable
    (e.g., Gaussian, Poisson, Negative Binomial). This determines the link function and variance structure.
- `solver`: An instance of [`AbstractLinearSolver`][] to use for solving the weighted least squares problem
    for inference.
"""
