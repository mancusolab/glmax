# pattern: Functional Core

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.stats as jaxstats

from jax.scipy import linalg as jscla
from jaxtyping import Array

from ._fit import FittedGLM
from .family.dist import Gaussian


T = TypeVar("T")


__all__ = [
    "AbstractDiagnostic",
    "DevianceResidual",
    "GoodnessOfFit",
    "GofStats",
    "Influence",
    "InfluenceStats",
    "PearsonResidual",
    "QuantileResidual",
    "check",
]


class AbstractDiagnostic(eqx.Module, Generic[T]):
    r"""Abstract base for pluggable GLM diagnostic strategies.

    Subclass and implement `diagnose` to define a diagnostic computation.
    Each concrete diagnostic encapsulates one computation and returns a typed
    result `T` (either a JAX array or an `eqx.Module` of arrays).
    """

    @abstractmethod
    def diagnose(self, fitted: FittedGLM) -> T:
        r"""Compute the diagnostic from a fitted GLM.

        !!! example
            ```python
            class MyDiag(AbstractDiagnostic[Array]):
                def diagnose(self, fitted: glmax.FittedGLM) -> Array:
                    return fitted.y - fitted.mu
            ```

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun produced by
          [`glmax.fit`][].

        **Returns:**

        Diagnostic result of type `T` - a JAX array or an `eqx.Module`
        containing only JAX arrays (pytree-compatible).
        """


class PearsonResidual(AbstractDiagnostic[Array], strict=True):
    r"""Pearson residuals $(y_i - \mu_i) / \sqrt{V(\mu_i)}$.

    These residuals normalize the raw residual $y_i - \mu_i$ by the square
    root of the family variance function $V(\mu_i)$, where $y_i$ is the
    observed response for observation $i$ and $\mu_i$ is the fitted mean.
    """

    def diagnose(self, fitted: FittedGLM) -> Array:
        r"""Compute Pearson residuals.

        The residual for observation $i$ is
        $r_i = (y_i - \mu_i) / \sqrt{V(\mu_i)}$, where $V(\mu_i)$ is the
        family variance function evaluated at the fitted mean $\mu_i$.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.

        **Returns:**

        Pearson residuals, shape `(n,)`.
        """
        family = fitted.family
        mu = fitted.mu
        v = family.variance(mu, fitted.params.disp, aux=fitted.params.aux)
        v = jnp.clip(jnp.asarray(v), min=jnp.finfo(float).tiny)
        return (fitted.y - mu) / jnp.sqrt(v)


class DevianceResidual(AbstractDiagnostic[Array], strict=True):
    r"""Deviance residuals $\operatorname{sign}(y_i - \mu_i) \sqrt{d_i}$.

    Here $y_i$ is the observed response, $\mu_i$ is the fitted mean, and
    $d_i$ is the deviance contribution for observation $i$.
    """

    def diagnose(self, fitted: FittedGLM) -> Array:
        r"""Compute deviance residuals.

        The residual for observation $i$ is
        $r_i = \operatorname{sign}(y_i - \mu_i)\sqrt{d_i}$, where $d_i$ is
        the per-observation deviance contribution.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.

        **Returns:**

        Deviance residuals, shape `(n,)`.
        """
        family = fitted.family
        y = fitted.y
        mu = fitted.mu
        d = family.deviance_contribs(y, mu, fitted.params.disp, aux=fitted.params.aux)
        d = jnp.clip(jnp.asarray(d), min=0.0)
        return jnp.sign(y - mu) * jnp.sqrt(d)


_EPS = jnp.finfo(jnp.float64).eps


class QuantileResidual(AbstractDiagnostic[Array], strict=True):
    r"""Deterministic quantile residuals via a mid-quantile approximation.

    For discrete families (Poisson, Binomial, NegativeBinomial) this uses
    $\Phi^{-1}((F(y_i) + F(y_i - 1))/2)$. For continuous families (Gaussian,
    Gamma) it uses $\Phi^{-1}(F(y_i))$. Here $F$ is the fitted cumulative
    distribution function and $\Phi^{-1}$ is the standard normal quantile
    function.

    CDF values are clamped to $[\varepsilon, 1-\varepsilon]$ before the
    normal quantile function to prevent infinite outputs, where
    $\varepsilon$ is machine epsilon for `float64`.
    """

    def diagnose(self, fitted: FittedGLM) -> Array:
        r"""Compute deterministic quantile residuals.

        For discrete responses this uses the mid-quantile correction
        $\Phi^{-1}((F(y_i) + F(y_i - 1))/2)$. For continuous responses this
        uses $\Phi^{-1}(F(y_i))$.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.

        **Returns:**

        Quantile residuals, shape `(n,)`.
        """
        family = fitted.family
        y = fitted.y
        mu = fitted.mu
        disp = fitted.params.disp
        aux = fitted.params.aux

        p_upper = family.cdf(y, mu, disp, aux=aux)
        p_lower = family.cdf(y - 1.0, mu, disp, aux=aux) if family.is_discrete else p_upper
        p_mid = 0.5 * (p_upper + p_lower)
        p_mid = jnp.clip(jnp.asarray(p_mid), _EPS, 1.0 - _EPS)
        return jaxstats.norm.ppf(p_mid)


class GofStats(eqx.Module, strict=True):
    r"""Goodness-of-fit statistics for a fitted GLM.

    All fields are scalar JAX arrays. Pytree-compatible.

    **Fields:**

    - `deviance`: total deviance $D = \sum_i d_i$, where $d_i$ is the
      deviance contribution for observation $i$.
    - `pearson_chi2`: Pearson chi-squared statistic
      $\chi^2 = \sum_i (y_i - \mu_i)^2 / V(\mu_i)$.
    - `df_resid`: residual degrees of freedom $n - p$, where $n$ is the
      number of observations and $p$ is the number of coefficients.
    - `dispersion`: fitted dispersion estimate $\hat{\phi}$.
    - `aic`: Akaike information criterion
      $\mathrm{AIC} = -2 \ell + 2p$, where $\ell$ is the fitted
      log-likelihood.
    - `bic`: Bayesian information criterion
      $\mathrm{BIC} = -2 \ell + p \log n$.
    """

    deviance: Array
    pearson_chi2: Array
    df_resid: Array
    dispersion: Array
    aic: Array
    bic: Array


class GoodnessOfFit(AbstractDiagnostic[GofStats], strict=True):
    r"""Goodness-of-fit summary diagnostic.

    Computes scalar summaries based on deviance, Pearson residual scale, and
    information criteria derived from the fitted model.
    """

    def diagnose(self, fitted: FittedGLM) -> GofStats:
        r"""Compute goodness-of-fit statistics.

        This computes $D$, $\chi^2$, $\hat{\phi}$, $\mathrm{AIC}$, and
        $\mathrm{BIC}$, where $D$ is total deviance, $\chi^2$ is the
        Pearson chi-squared statistic, and $\hat{\phi}$ is the fitted
        dispersion.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.

        **Returns:**

        `GofStats` with scalar array fields.
        """
        family = fitted.family
        y = fitted.y
        mu = fitted.mu
        eta = fitted.eta
        disp = fitted.params.disp
        aux = fitted.params.aux
        n, p = fitted.X.shape

        deviance = jnp.sum(family.deviance_contribs(y, mu, disp, aux=aux))
        pearson_chi2 = jnp.sum((y - mu) ** 2 / family.variance(mu, disp, aux=aux))

        n_f = jnp.asarray(n, dtype=jnp.float64)
        p_f = jnp.asarray(p, dtype=jnp.float64)
        df_resid = n_f - p_f
        ll_disp = jnp.clip(deviance / n_f, min=jnp.finfo(float).tiny) if isinstance(family, Gaussian) else disp
        ll = -fitted.family.negloglikelihood(y, eta, ll_disp, aux=aux)
        aic = -2.0 * ll + 2.0 * p_f
        bic = -2.0 * ll + p_f * jnp.log(n_f)

        return GofStats(
            deviance=deviance,
            pearson_chi2=pearson_chi2,
            df_resid=df_resid,
            dispersion=jnp.asarray(disp),
            aic=aic,
            bic=bic,
        )


class InfluenceStats(eqx.Module, strict=True):
    r"""Per-observation influence statistics.

    **Fields:**

    - `leverage`: hat-matrix diagonal $h_{ii} \in (0, 1)$, shape `(n,)`.
    - `cooks_distance`: Cook's distance $D_i \geq 0$, shape `(n,)`, where
      $D_i$ measures the influence of observation $i$ on the fitted
      coefficient vector.
    """

    leverage: Array
    cooks_distance: Array


class Influence(AbstractDiagnostic[InfluenceStats], strict=True):
    r"""Leverage and Cook's distance via Cholesky-based hat-matrix computation.

    Recomputes $\operatorname{chol}(X^\top W X)$ from the fitted weights,
    where $X$ is the design matrix and $W$ is the diagonal matrix of working
    weights. It does not rely on the Cholesky factor from IRLS because that
    factor is not persisted in [`glmax.FitResult`][].
    """

    def diagnose(self, fitted: FittedGLM) -> InfluenceStats:
        r"""Compute leverage and Cook's distance.

        The leverage values are the diagonal elements $h_{ii}$ of the hat
        matrix. Cook's distance is computed from $h_{ii}$, the Pearson
        residual, and the coefficient count $p$.

        **Arguments:**

        - `fitted`: fitted [`glmax.FittedGLM`][] noun.

        **Returns:**

        `InfluenceStats` with `leverage` and `cooks_distance`, each shape `(n,)`.
        """
        X = fitted.X
        y = fitted.y
        mu = fitted.mu
        w = fitted.glm_wt
        disp = fitted.params.disp
        aux = fitted.params.aux
        _, p = X.shape

        sqrt_w = jnp.sqrt(w)
        Xw = X * sqrt_w[:, jnp.newaxis]
        A = Xw.T @ Xw
        L = jscla.cholesky(A, lower=True)
        Z = jscla.solve_triangular(L, Xw.T, lower=True)
        leverage = jnp.sum(Z**2, axis=0)

        v = fitted.family.variance(mu, disp, aux=aux)
        r_pearson = (y - mu) / jnp.sqrt(v)
        p_f = jnp.asarray(p, dtype=jnp.float64)
        cooks_distance = r_pearson**2 * leverage / (p_f * (1.0 - leverage) ** 2)

        return InfluenceStats(leverage=leverage, cooks_distance=cooks_distance)


@eqx.filter_jit
def check(
    fitted: FittedGLM,
    *,
    diagnostic: AbstractDiagnostic[T] = GoodnessOfFit(),
) -> T:
    r"""Assess model fit with one diagnostic and return its typed result.

    The canonical `check` grammar verb. Accepts one concrete
    `AbstractDiagnostic[T]` instance and returns the corresponding result
    `T = diagnostic.diagnose(fitted)`.

    Decorated with `eqx.filter_jit`; JIT-compiles on first call and caches
    subsequent calls with the same structure.

    !!! example "Compute multiple diagnostics with `tree_map`"
        ```python
        import jax.tree_util as jtu
        import glmax

        diagnostics = (
            glmax.PearsonResidual(),
            glmax.DevianceResidual(),
            glmax.GoodnessOfFit(),
        )

        results = jtu.tree_map(
            lambda diagnostic: glmax.check(fitted, diagnostic=diagnostic),
            diagnostics,
            is_leaf=lambda node: isinstance(node, glmax.AbstractDiagnostic),
        )
        pearson, deviance, gof = results
        ```

    **Arguments:**

    - `fitted`: fitted [`glmax.FittedGLM`][] noun produced by
      [`glmax.fit`][].
    - `diagnostic`: [`glmax.AbstractDiagnostic`][] instance to apply.
      Defaults to [`glmax.GoodnessOfFit`][].

    **Returns:**

    One diagnostic result of type `T`.

    **Raises:**

    - `TypeError`: if `fitted` is not a [`glmax.FittedGLM`][] instance or
      `diagnostic` is not a [`glmax.AbstractDiagnostic`][] instance.
    """
    if not isinstance(fitted, FittedGLM):
        raise TypeError(f"check(...) expects `fitted` to be a FittedGLM instance, got {type(fitted).__name__!r}.")
    if not isinstance(diagnostic, AbstractDiagnostic):
        raise TypeError(
            f"check(...) expects `diagnostic` to be an AbstractDiagnostic instance, got {type(diagnostic).__name__!r}."
        )

    return diagnostic.diagnose(fitted)
