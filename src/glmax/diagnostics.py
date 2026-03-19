# pattern: Functional Core

from abc import abstractmethod
from typing import Generic, NamedTuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.stats as jaxstats

from jaxtyping import Array

from ._fit import FittedGLM
from .family.dist import Binomial, NegativeBinomial, Poisson


T = TypeVar("T")


__all__ = [
    "AbstractDiagnostic",
    "Diagnostics",
    "PearsonResidual",
    "DevianceResidual",
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


class Diagnostics(NamedTuple):
    """Model-fit diagnostics contract returned by `check(...)`.

    !!! note
        This is a placeholder contract. No diagnostic fields are computed yet.
        The seam is reserved for residuals, calibration, and influence diagnostics
        in a future release.
    """


def check(fitted: FittedGLM) -> Diagnostics:
    r"""Assess model fit and return a diagnostics noun.

    The canonical `check` grammar verb. Currently returns an empty `Diagnostics`
    placeholder; use as a seam for residual and calibration diagnostics.

    **Arguments:**

    - `fitted`: `FittedGLM` noun produced by `fit(...)`.

    **Returns:**

    `Diagnostics` noun (currently empty).

    **Raises:**

    - `TypeError`: if `fitted` is not a `FittedGLM` instance.
    """
    if not isinstance(fitted, FittedGLM):
        raise TypeError("check(...) expects `fitted` to be a FittedGLM instance.")

    return Diagnostics()
