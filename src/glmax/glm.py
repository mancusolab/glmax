from typing import Tuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike, ScalarLike

from .contracts import Diagnostics, FitResult, Params
from .family.dist import ExponentialFamily, Gaussian, NegativeBinomial, Poisson
from .family.utils import t_cdf
from .infer.optimize import irls
from .infer.solve import AbstractLinearSolver, CholeskySolver
from .infer.stderr import AbstractStdErrEstimator, FisherInfoError


# Backward-compatible alias while transitioning callers to the canonical contract noun.
GLMState = FitResult


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

        -  A [`glmax.FitResult`][] containing the final estimated parameters and convergence diagnostics
            from the fitted GLM model.
        """
        if init is None or alpha_init is None:
            init, alpha_init = self.calc_eta_and_dispersion(X, y, offset_eta)

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
            params=Params(beta=beta, disp=alpha),
            se=beta_se,
            z=stat,
            p=pval_wald,
            eta=eta,
            mu=mu,
            glm_wt=weight,
            diagnostics=Diagnostics(converged=converged, num_iters=n_iter),
            infor_inv=resid_covar,
            resid=resid,
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
