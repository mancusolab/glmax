# GLM Diagnostics Implementation Plan — Phase 2

**Goal:** Add `AbstractDiagnostic[T]`, `PearsonResidual`, `DevianceResidual`, and `QuantileResidual` to `src/glmax/diagnostics.py`.

**Architecture:** Single file, coherent module. New classes appended to the existing `diagnostics.py`; the old `check()` stub and `Diagnostics` NamedTuple remain untouched until Phase 4. Concrete residual classes are `eqx.Module` with `strict=True`; the abstract base is `eqx.Module, Generic[T]` without `strict=True`, matching the `ExponentialDispersionFamily` pattern.

**Tech Stack:** JAX, Equinox, `jax.scipy.stats`, `abc.abstractmethod`

**Scope:** Phase 2 of 4 — depends on Phase 1 (family `cdf` and `deviance_contribs`)

**Codebase verified:** 2026-03-19

---

## Acceptance Criteria Coverage

### glm-diagnostics.AC1: AbstractDiagnostic base class
- **glm-diagnostics.AC1.1 Success:** `AbstractDiagnostic` is an `eqx.Module` subclass parameterised by `Generic[T]` with abstract method `diagnose(fitted: FittedGLM) -> T`
- **glm-diagnostics.AC1.2 Success:** Concrete subclasses that implement `diagnose` can be instantiated and passed to `check()`
- **glm-diagnostics.AC1.3 Failure:** Instantiating `AbstractDiagnostic` directly (without subclassing) raises an error

### glm-diagnostics.AC2: PearsonResidual
- **glm-diagnostics.AC2.1 Success:** Returns array of shape `(n,)` equal to `(y - mu) / sqrt(V(mu))` matching statsmodels `resid_pearson` for Gaussian, Poisson, and Binomial families
- **glm-diagnostics.AC2.2 Edge:** Produces correct values when `mu` is close to 0 or to the boundary of the family's support (no NaN under JIT)

### glm-diagnostics.AC3: DevianceResidual
- **glm-diagnostics.AC3.1 Success:** Returns array of shape `(n,)` equal to `sign(y - mu) * sqrt(deviance_contribution_i)` matching statsmodels `resid_deviance` for Gaussian, Poisson, and Binomial families
- **glm-diagnostics.AC3.2 Edge:** Poisson with `y=0` produces a finite (non-NaN) value using `0 * log(0/mu) = 0` convention

### glm-diagnostics.AC4: QuantileResidual
- **glm-diagnostics.AC4.1 Success:** Returns array of shape `(n,)` that is finite for all observations across all five families
- **glm-diagnostics.AC4.2 Success:** For Gaussian family, quantile residuals equal standardised Pearson residuals (exact, since CDF is the normal CDF)
- **glm-diagnostics.AC4.3 Edge:** CDF values at 0 or 1 boundaries are clamped before `norm.ppf`; output is finite (not ±inf)
- **glm-diagnostics.AC4.4 Success:** Deterministic — same `FittedGLM` always produces the same quantile residuals with no PRNG key required

---

## Key Codebase Facts

- `FittedGLM` fields (via properties): `y`, `X`, `mu`, `eta`, `params` (Params with `.beta`, `.disp`, `.aux`), `glm_wt`, `score_residual`, `converged`, `num_iters`
- `FittedGLM.model` is a `GLM` with `.family` (an `ExponentialDispersionFamily`)
- Variance: `fitted.model.family.variance(mu, disp, aux)` — returns shape `(n,)` — already abstract on base class
- Deviance contribs: `fitted.model.family.deviance_contribs(y, mu, disp, aux)` — from Phase 1
- CDF: `fitted.model.family.cdf(y, mu, disp, aux)` — from Phase 1
- Discrete families: `Poisson`, `Binomial`, `NegativeBinomial` (mid-quantile uses `cdf(y-1)`); continuous: `Gaussian`, `Gamma` (use `cdf(y)` directly)
- `isinstance` check for discrete is static at trace time (safe under `eqx.filter_jit`)
- Existing `diagnostics.py` imports: `from ._fit import FittedGLM` — keep all existing imports
- Existing test: `tests/diagnostics/test_check.py` — must still pass (old `check()` and `Diagnostics` unchanged in this phase)
- New tests: `tests/diagnostics/test_residuals.py`
- Run tests: `pytest -p no:capture tests`

---

<!-- START_SUBCOMPONENT_A (tasks 1-4) -->

<!-- START_TASK_1 -->
### Task 1: Add `AbstractDiagnostic[T]` to `diagnostics.py`

**Verifies:** glm-diagnostics.AC1.1, glm-diagnostics.AC1.3

**Files:**
- Modify: `src/glmax/diagnostics.py`

**Implementation:**

Prepend these imports and the base class to `diagnostics.py`, before the existing `Diagnostics` NamedTuple. Do NOT remove the existing stub yet.

At the top of the file, add the new imports alongside the existing ones:

```python
# pattern: Functional Core

from abc import abstractmethod
from typing import Generic, NamedTuple, TypeVar

import equinox as eqx

from jaxtyping import Array

from ._fit import FittedGLM


T = TypeVar("T")


__all__ = [
    "AbstractDiagnostic",
    "Diagnostics",  # kept until Phase 4 removes it
    "check",
]


class AbstractDiagnostic(eqx.Module, Generic[T]):
    r"""Abstract base for pluggable GLM diagnostic strategies.

    Subclass and implement `diagnose` to define a diagnostic computation.
    Each concrete diagnostic encapsulates one computation and returns a
    typed result `T` (either a JAX array or an `eqx.Module` of arrays).

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

        Diagnostic result of type `T` — a JAX array or an `eqx.Module`
        containing only JAX arrays (pytree-compatible).
        """
```

Then keep the existing `Diagnostics` and `check()` below unchanged.

**Testing:**

First, create `tests/diagnostics/conftest.py` with shared fit helpers (used by test_residuals.py, test_gof.py, and test_influence.py to avoid duplication):

```python
# pattern: Imperative Shell
"""Shared fixture helpers for diagnostics tests."""

import numpy as np
import pytest

import jax.numpy as jnp

import glmax
from glmax import GLMData


def fit_gaussian():
    X_raw = np.array([[1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.0, 2.0], [1.0, 2.5]])
    y_raw = np.array([1.2, 1.8, 2.5, 3.1, 3.8])
    model = glmax.specify(family=glmax.Gaussian())
    data = GLMData(X=jnp.array(X_raw), y=jnp.array(y_raw))
    return glmax.fit(model, data), X_raw, y_raw


def fit_poisson():
    X_raw = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 1.5]])
    y_raw = np.array([1.0, 2.0, 4.0, 7.0, 3.0])
    model = glmax.specify(family=glmax.Poisson())
    data = GLMData(X=jnp.array(X_raw), y=jnp.array(y_raw))
    return glmax.fit(model, data), X_raw, y_raw


def fit_binomial():
    X_raw = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, -0.5]])
    y_raw = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    model = glmax.specify(family=glmax.Binomial())
    data = GLMData(X=jnp.array(X_raw), y=jnp.array(y_raw))
    return glmax.fit(model, data), X_raw, y_raw
```

Then create `tests/diagnostics/test_residuals.py`:

```python
# pattern: Imperative Shell

import pytest

import jax.numpy as jnp

import glmax
import glmax.diagnostics
from glmax import GLMData
from glmax.diagnostics import AbstractDiagnostic
from glmax.family import Gaussian

from .conftest import fit_gaussian, fit_poisson, fit_binomial


class _ConcreteOK(AbstractDiagnostic):
    def diagnose(self, fitted):
        return fitted.y


def test_abstract_diagnostic_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AbstractDiagnostic()


def test_concrete_subclass_can_be_instantiated():
    d = _ConcreteOK()
    assert isinstance(d, AbstractDiagnostic)
```

**Verification:**

Run: `pytest -p no:capture tests/diagnostics/`
Expected: Both old `test_check.py` tests pass; new AC1 tests pass.

**Commit:** `feat(diagnostics): add AbstractDiagnostic[T] base class`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Implement `PearsonResidual`

**Verifies:** glm-diagnostics.AC2.1, glm-diagnostics.AC2.2

**Files:**
- Modify: `src/glmax/diagnostics.py` (add after `AbstractDiagnostic`)
- Modify: `src/glmax/diagnostics.py` `__all__`
- Test: `tests/diagnostics/test_residuals.py`

**Implementation:**

Add `PearsonResidual` to `diagnostics.py` after `AbstractDiagnostic`. Formula: `(y - mu) / sqrt(V(mu))`.

```python
import jax.numpy as jnp

from jaxtyping import Array


class PearsonResidual(AbstractDiagnostic[Array], strict=True):
    r"""Pearson residuals $(y_i - \mu_i) / \sqrt{V(\mu_i)}$.

    Residuals normalised by the square root of the family's variance function.
    Standard normal under the true model for large $n$.
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
        disp = fitted.params.disp
        aux = fitted.params.aux
        v = family.variance(mu, disp, aux=aux)
        return (fitted.y - mu) / jnp.sqrt(v)
```

Add `"PearsonResidual"` to `__all__`.

**Testing** (add to `tests/diagnostics/test_residuals.py`):

Use the shared helpers from `tests/diagnostics/conftest.py` (created in Task 1). Add `import statsmodels.api as sm` and `import numpy as np` to the imports at the top of `test_residuals.py`.

```python
class TestPearsonResidual:
    def test_pearson_shape_n(self):
        fitted, _, _ = fit_gaussian()
        result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert result.shape == (5,)

    def test_pearson_gaussian_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_gaussian()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Gaussian()).fit()
        expected = sm_result.resid_pearson
        result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_pearson_poisson_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_poisson()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Poisson()).fit()
        expected = sm_result.resid_pearson
        result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_pearson_binomial_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_binomial()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Binomial()).fit()
        expected = sm_result.resid_pearson
        result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(expected), atol=1e-5)

    def test_pearson_all_finite(self):
        for fit_fn in [fit_gaussian, fit_poisson, fit_binomial]:
            fitted, _, _ = fit_fn()
            result = glmax.diagnostics.PearsonResidual().diagnose(fitted)
            assert jnp.all(jnp.isfinite(result))
```

Note: add `import glmax.diagnostics` to the imports at the top of the test file.

**Verification:**

Run: `pytest -p no:capture tests/diagnostics/`
Expected: All tests pass

**Commit:** `feat(diagnostics): add PearsonResidual`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Implement `DevianceResidual`

**Verifies:** glm-diagnostics.AC3.1, glm-diagnostics.AC3.2

**Files:**
- Modify: `src/glmax/diagnostics.py` (add after `PearsonResidual`)
- Modify: `src/glmax/diagnostics.py` `__all__`
- Test: `tests/diagnostics/test_residuals.py`

**Implementation:**

Formula: `sign(y - mu) * sqrt(deviance_contribs_i)`. The `deviance_contribs` method (Phase 1) already handles `y=0` for Poisson with the `0*log(0/mu)=0` convention.

```python
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
        disp = fitted.params.disp
        aux = fitted.params.aux
        d = family.deviance_contribs(y, mu, disp, aux=aux)
        return jnp.sign(y - mu) * jnp.sqrt(d)
```

Add `"DevianceResidual"` to `__all__`.

**Testing** (add to `tests/diagnostics/test_residuals.py`):

```python
class TestDevianceResidual:
    def test_deviance_shape_n(self):
        fitted, _, _ = fit_gaussian()
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert result.shape == (5,)

    def test_deviance_gaussian_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_gaussian()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Gaussian()).fit()
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(sm_result.resid_deviance), atol=1e-5)

    def test_deviance_poisson_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_poisson()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Poisson()).fit()
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(sm_result.resid_deviance), atol=1e-5)

    def test_deviance_binomial_matches_statsmodels(self):
        fitted, X_raw, y_raw = fit_binomial()
        sm_result = sm.GLM(y_raw, X_raw, family=sm.families.Binomial()).fit()
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert jnp.allclose(result, jnp.array(sm_result.resid_deviance), atol=1e-5)

    def test_deviance_poisson_zero_y_finite(self):
        # Fit with a y=0 observation
        X_raw = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y_raw = np.array([0.0, 1.0, 2.0, 4.0, 6.0])
        model = glmax.specify(family=glmax.Poisson())
        data = GLMData(X=jnp.array(X_raw), y=jnp.array(y_raw))
        fitted = glmax.fit(model, data)
        result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_deviance_all_finite(self):
        for fit_fn in [fit_gaussian, fit_poisson, fit_binomial]:
            fitted, _, _ = fit_fn()
            result = glmax.diagnostics.DevianceResidual().diagnose(fitted)
            assert jnp.all(jnp.isfinite(result))
```

**Verification:**

Run: `pytest -p no:capture tests/diagnostics/`
Expected: All tests pass

**Commit:** `feat(diagnostics): add DevianceResidual`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Implement `QuantileResidual`

**Verifies:** glm-diagnostics.AC4.1, glm-diagnostics.AC4.2, glm-diagnostics.AC4.3, glm-diagnostics.AC4.4

**Files:**
- Modify: `src/glmax/diagnostics.py` (add after `DevianceResidual`)
- Modify: `src/glmax/diagnostics.py` `__all__`
- Test: `tests/diagnostics/test_residuals.py`

**Implementation:**

Mid-quantile approximation: for discrete families `(Poisson, Binomial, NegativeBinomial)` use `(cdf(y) + cdf(y-1)) / 2`; for continuous `(Gaussian, Gamma)` use `cdf(y)`. The `isinstance` check is evaluated at Python trace time (not inside JAX trace), so it's JIT-safe.

CDF output is clamped to `[eps, 1-eps]` before `norm.ppf` to prevent ±inf.

```python
import jax

from glmax.family.dist import Binomial, NegativeBinomial, Poisson

_DISCRETE_FAMILIES = (Poisson, Binomial, NegativeBinomial)
_EPS = jnp.finfo(jnp.float64).eps


class QuantileResidual(AbstractDiagnostic[Array], strict=True):
    r"""Deterministic quantile residuals via the mid-quantile approximation.

    For discrete families (Poisson, Binomial, NegativeBinomial) uses
    $\Phi^{-1}\!\left(\tfrac{F(y) + F(y-1)}{2}\right)$.
    For continuous families (Gaussian, Gamma) uses $\Phi^{-1}(F(y))$.

    CDF values are clamped to $[\varepsilon, 1-\varepsilon]$ before the
    normal quantile function to prevent $\pm\infty$ output.
    """

    def diagnose(self, fitted: FittedGLM) -> Array:
        r"""Compute deterministic quantile residuals.

        **Arguments:**

        - `fitted`: `FittedGLM` produced by `fit(...)`.

        **Returns:**

        Quantile residuals, shape `(n,)`. Standard normal under the true model.
        """
        family = fitted.model.family
        y = fitted.y
        mu = fitted.mu
        disp = fitted.params.disp
        aux = fitted.params.aux

        p_upper = family.cdf(y, mu, disp, aux=aux)
        if isinstance(family, _DISCRETE_FAMILIES):
            p_lower = family.cdf(y - 1.0, mu, disp, aux=aux)
        else:
            p_lower = p_upper
        p_mid = (p_upper + p_lower) / 2.0
        p_clamped = jnp.clip(p_mid, _EPS, 1.0 - _EPS)
        return jax.scipy.stats.norm.ppf(p_clamped)
```

Add `"QuantileResidual"` to `__all__`.

Note: the import `from glmax.family.dist import Binomial, NegativeBinomial, Poisson` at the module level in `diagnostics.py` is fine — `diagnostics.py` already imports from `._fit`, and `family.dist` is a sibling subpackage. Use a relative import: `from .family.dist import Binomial, NegativeBinomial, Poisson`.

**Testing** (add to `tests/diagnostics/test_residuals.py`):

```python
class TestQuantileResidual:
    def test_quantile_shape_n(self):
        fitted, _, _ = fit_gaussian()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert result.shape == (5,)

    def test_quantile_gaussian_all_finite(self):
        fitted, _, _ = fit_gaussian()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_gaussian_equals_standardised_pearson(self):
        # For Gaussian with identity link: F(y|mu,sigma) = Phi((y-mu)/sigma)
        # Phi^-1(F(y)) = (y-mu)/sigma
        # PearsonResidual = (y-mu)/sqrt(V(mu)) = (y-mu)/sigma (since V(mu)=sigma^2 for Gaussian)
        # So quantile residual == Pearson residual exactly
        fitted, X_raw, y_raw = fit_gaussian()
        q = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        p = glmax.diagnostics.PearsonResidual().diagnose(fitted)
        assert jnp.allclose(q, p, atol=1e-5)

    def test_quantile_poisson_all_finite(self):
        fitted, _, _ = fit_poisson()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_binomial_all_finite(self):
        fitted, _, _ = fit_binomial()
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))

    def test_quantile_deterministic(self):
        # Same FittedGLM should produce same output
        fitted, _, _ = fit_gaussian()
        r1 = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        r2 = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.allclose(r1, r2, atol=0.0)

    def test_quantile_poisson_zero_y_finite(self):
        X_raw = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y_raw = np.array([0.0, 1.0, 2.0, 4.0, 6.0])
        model = glmax.specify(family=glmax.Poisson())
        data = GLMData(X=jnp.array(X_raw), y=jnp.array(y_raw))
        fitted = glmax.fit(model, data)
        result = glmax.diagnostics.QuantileResidual().diagnose(fitted)
        assert jnp.all(jnp.isfinite(result))
```

**Verification:**

Run: `pytest -p no:capture tests/diagnostics/`
Expected: All tests pass

Run: `pytest -p no:capture tests`
Expected: All 321+ tests pass

**Commit:** `feat(diagnostics): add QuantileResidual with mid-quantile approximation`
<!-- END_TASK_4 -->

<!-- END_SUBCOMPONENT_A -->
