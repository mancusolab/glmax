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
        """Calculate eta and canonical dispersion value.

        :param X: covariate data matrix (nxp)
        :param y: outcome vector (nx1)
        :param offset_eta: offset (nx1)
        :return: eta and canonical dispersion value
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
        X: ArrayLike | GLMData,
        y: ArrayLike | None = None,
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
        - `alpha_init`: initial value for the dispersion parameter, default to 0s.

        **Returns:**

        -  A [`glmax.FitResult`][] containing the final estimated parameters and convergence diagnostics
            from the fitted GLM model.
        """
        data = self._coerce_data(X, y, offset_eta)
        if data.weights is not None:
            raise ValueError("GLMData.weights is not supported in GLM.fit yet.")
        X_array, y_array, offset_array, _, _ = data.canonical_arrays()
        if X_array.shape[0] == 0:
            raise ValueError("GLMData.mask removes all samples; at least one effective sample is required.")

        disp_init = alpha_init
        if init is not None:
            init = jnp.asarray(init)
            if init.ndim != 1 or init.shape[0] != X_array.shape[0]:
                raise ValueError("init must be a one-dimensional eta vector with length equal to sample count.")
            if not bool(jnp.all(jnp.isfinite(init))):
                raise ValueError("init must contain only finite values.")

        if disp_init is not None:
            disp_init = jnp.asarray(disp_init)
            if disp_init.ndim > 0 and disp_init.size != 1:
                raise ValueError("alpha_init must be a scalar dispersion value.")
            if not bool(jnp.all(jnp.isfinite(disp_init))):
                raise ValueError("alpha_init must contain only finite values.")

        if init is None or disp_init is None:
            init, disp_init = self.calc_eta_and_dispersion(X_array, y_array, offset_array)

        beta, n_iter, converged, disp = irls(
            X_array,
            y_array,
            self.family,
            self.solver,
            init,
            max_iter,
            tol,
            step_size,
            offset_array,
            disp_init=disp_init,
        )

        eta = X_array @ beta + offset_array
        mu = self.family.glink.inverse(eta)
        resid = (y_array - mu) * self.family.glink.deriv(mu)  # note: this is the working resid

        _, _, weight = self.family.calc_weight(X_array, y_array, eta, disp)

        resid_covar = se_estimator(self.family, X_array, y_array, eta, mu, weight, disp)
        beta_se = jnp.sqrt(jnp.diag(resid_covar))

        df = X_array.shape[0] - X_array.shape[1]
        beta = jnp.ravel(beta)  # (p,)
        stat = beta / beta_se

        pval_wald = self.wald_test(stat, df)

        return GLMState(
            params=Params(beta=beta, disp=self.family.canonical_dispersion(disp)),
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

    @staticmethod
    def _coerce_data(X: ArrayLike | GLMData, y: ArrayLike | None, offset_eta: ArrayLike) -> GLMData:
        if isinstance(X, GLMData):
            if y is not None:
                raise TypeError("When GLMData is provided as X, y must be omitted.")
            offset_array = jnp.asarray(offset_eta)
            if not bool(jnp.all(offset_array == 0)):
                raise TypeError("offset_eta must remain default when GLMData is provided.")
            return X

        if y is None:
            raise TypeError("y is required when X is not a GLMData object.")

        return GLMData(X=X, y=y, offset=offset_eta)


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
