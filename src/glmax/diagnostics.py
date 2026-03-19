# pattern: Functional Core

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.stats as jaxstats

from jax.scipy import linalg as jscla
from jaxtyping import Array

from ._fit import FittedGLM
from .family.dist import Binomial, Gaussian, NegativeBinomial, Poisson


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

    **Example:**

    ```python
    class MyDiag(AbstractDiagnostic[Array]):
        def diagnose(self, fitted: FittedGLM) -> Array:
            return fitted.y - fitted.mu
    ```
    """

    @abstractmethod
    def diagnose(self, fitted: FittedGLM) -> T:
        r"""Compute the diagnostic from a fitted GLM.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

        **Returns:**

        Diagnostic result of type `T` - a JAX array or an `eqx.Module`
        containing only JAX arrays (pytree-compatible).
        """


class PearsonResidual(AbstractDiagnostic[Array], strict=True):
    r"""Pearson residuals $(y_i - \mu_i) / \sqrt{V(\mu_i)}$.

    Residuals normalized by the square root of the family's variance function.
    """

    def diagnose(self, fitted: FittedGLM) -> Array:
        r"""Compute Pearson residuals.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

        **Returns:**

        Pearson residuals, shape `(n,)`.
        """
        family = fitted.model.family
        mu = fitted.mu
        v = family.variance(mu, fitted.params.disp, aux=fitted.params.aux)
        v = jnp.clip(jnp.asarray(v), min=jnp.finfo(float).tiny)
        return (fitted.y - mu) / jnp.sqrt(v)


class DevianceResidual(AbstractDiagnostic[Array], strict=True):
    r"""Deviance residuals $\operatorname{sign}(y_i - \mu_i) \sqrt{d_i}$.

    Signed square-root of each observation's deviance contribution.
    """

    def diagnose(self, fitted: FittedGLM) -> Array:
        r"""Compute deviance residuals.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

        **Returns:**

        Deviance residuals, shape `(n,)`.
        """
        family = fitted.model.family
        y = fitted.y
        mu = fitted.mu
        d = family.deviance_contribs(y, mu, fitted.params.disp, aux=fitted.params.aux)
        d = jnp.clip(jnp.asarray(d), min=0.0)
        return jnp.sign(y - mu) * jnp.sqrt(d)


_DISCRETE_FAMILIES = (Poisson, Binomial, NegativeBinomial)
_EPS = jnp.finfo(jnp.float64).eps


class QuantileResidual(AbstractDiagnostic[Array], strict=True):
    r"""Deterministic quantile residuals via a mid-quantile approximation.

    For discrete families (Poisson, Binomial, NegativeBinomial) this uses
    $\Phi^{-1}((F(y) + F(y-1))/2)$. For continuous families (Gaussian, Gamma)
    it uses $\Phi^{-1}(F(y))$.

    CDF values are clamped to $[\varepsilon, 1-\varepsilon]$ before the
    normal quantile function to prevent infinite outputs.
    """

    def diagnose(self, fitted: FittedGLM) -> Array:
        r"""Compute deterministic quantile residuals.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

        **Returns:**

        Quantile residuals, shape `(n,)`.
        """
        family = fitted.model.family
        y = fitted.y
        mu = fitted.mu
        disp = fitted.params.disp
        aux = fitted.params.aux

        p_upper = family.cdf(y, mu, disp, aux=aux)
        p_lower = family.cdf(y - 1.0, mu, disp, aux=aux) if isinstance(family, _DISCRETE_FAMILIES) else p_upper
        p_mid = 0.5 * (p_upper + p_lower)
        p_mid = jnp.clip(jnp.asarray(p_mid), _EPS, 1.0 - _EPS)
        return jaxstats.norm.ppf(p_mid)


class GofStats(eqx.Module, strict=True):
    r"""Goodness-of-fit statistics for a fitted GLM.

    All fields are scalar JAX arrays. Pytree-compatible.

    **Fields:**

    - `deviance`: total deviance $D = \sum_i d_i$.
    - `pearson_chi2`: Pearson chi-squared $\sum_i (y_i - \mu_i)^2 / V(\mu_i)$.
    - `df_resid`: residual degrees of freedom $n - p$.
    - `dispersion`: fitted dispersion parameter $\hat\phi$.
    - `aic`: Akaike information criterion $-2\ell + 2p$.
    - `bic`: Bayesian information criterion $-2\ell + p \log n$.
    """

    deviance: Array
    pearson_chi2: Array
    df_resid: Array
    dispersion: Array
    aic: Array
    bic: Array


class GoodnessOfFit(AbstractDiagnostic[GofStats], strict=True):
    r"""Goodness-of-fit statistics: deviance, Pearson chi-squared, AIC, BIC, dispersion."""

    def diagnose(self, fitted: FittedGLM) -> GofStats:
        r"""Compute goodness-of-fit statistics.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

        **Returns:**

        `GofStats` with scalar array fields.
        """
        family = fitted.model.family
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
        ll = fitted.model.log_prob(y, eta, ll_disp, aux=aux)
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
    - `cooks_distance`: Cook's distance $D_i \geq 0$, shape `(n,)`.
    """

    leverage: Array
    cooks_distance: Array


class Influence(AbstractDiagnostic[InfluenceStats], strict=True):
    r"""Leverage and Cook's distance via Cholesky-based hat-matrix computation.

    Recomputes $\operatorname{chol}(X^T W X)$ from the fitted weights;
    does not rely on the Cholesky factor from IRLS (which is not persisted).
    """

    def diagnose(self, fitted: FittedGLM) -> InfluenceStats:
        r"""Compute leverage and Cook's distance.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

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

        v = fitted.model.family.variance(mu, disp, aux=aux)
        r_pearson = (y - mu) / jnp.sqrt(v)
        p_f = jnp.asarray(p, dtype=jnp.float64)
        cooks_distance = r_pearson**2 * leverage / (p_f * (1.0 - leverage) ** 2)

        return InfluenceStats(leverage=leverage, cooks_distance=cooks_distance)


@eqx.filter_jit
def check(
    fitted: FittedGLM,
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

    - `fitted`: `FittedGLM` noun produced by `fit(...)`.
    - `diagnostic`: `AbstractDiagnostic[T]` to apply. Defaults to
      `PearsonResidual()`.

    **Returns:**

    One diagnostic result of type `T`.

    **Raises:**

    - `TypeError`: if `fitted` is not a `FittedGLM` instance.
    """
    if not isinstance(fitted, FittedGLM):
        raise TypeError(f"check(...) expects `fitted` to be a FittedGLM instance, got {type(fitted).__name__!r}.")
    if not isinstance(diagnostic, AbstractDiagnostic):
        raise TypeError(
            f"check(...) expects `diagnostic` to be an AbstractDiagnostic instance, got {type(diagnostic).__name__!r}."
        )

    return diagnostic.diagnose(fitted)
