from typing import NamedTuple

import equinox as eqx

from jax import Array, numpy as jnp
from jax.scipy.stats import norm
from jaxtyping import ArrayLike, ScalarLike

from src.glmax.family.distribution import ExponentialFamily, Gaussian
from src.glmax.family.utils import t_cdf
from src.glmax.infer.optimize import irls
from src.glmax.infer.solve import AbstractLinearSolver, CholeskySolver
from src.glmax.infer.stderr import AbstractStdErrEstimator, FisherInfoError


class GLMState(NamedTuple):
    """
    TODO: documentation; what are these things?
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
    """TODO: add better documentation here."""

    family: ExponentialFamily = Gaussian()
    solver: AbstractLinearSolver = CholeskySolver()

    def wald_test(self, statistic: ArrayLike, df: int) -> Array:
        """
        beta_MLE ~ N(beta, I^-1), for large sample size
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
        """TODO: add better documentation here.
        Fit GLM

        **Arguments:**

        - `X`: covariate data matrix (nxp)
        - `y`: outcome vector (nx1)
        - `offset_eta`: offset (nx1)
        - `init`: initial value for betas
        - `alpha_init`: initial value for alpha in NB model, default to 0s
        - `se_estimator`: estimator for standard error, default to fisher information

        **Returns:**

        A [`glmax.GLMState`][] object that contains model fitting result.
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
