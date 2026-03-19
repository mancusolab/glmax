# GLM Diagnostics Implementation Plan — Phase 4

**Goal:** Wire `check()` with `eqx.filter_jit`, define `DEFAULT_DIAGNOSTICS`, remove the old stub, and export everything through the public API.

**Architecture:** Replace the old `check()` stub and `Diagnostics` NamedTuple in `diagnostics.py` with the new implementations. Update `__init__.py` exports. Update existing `test_check.py` to the new API.

**Tech Stack:** JAX, Equinox (`eqx.filter_jit`), Python standard library

**Scope:** Phase 4 of 4 — depends on Phases 1–3

**Codebase verified:** 2026-03-19

---

## Acceptance Criteria Coverage

### glm-diagnostics.AC7: check() function
- **glm-diagnostics.AC7.1 Success:** `glmax.check(fitted)` with default diagnostics returns a 5-tuple mapping positionally to `(PearsonResidual, DevianceResidual, QuantileResidual, GoodnessOfFit, Influence)` results
- **glm-diagnostics.AC7.2 Success:** `glmax.check(fitted, diagnostics=(PearsonResidual(),))` returns a 1-tuple containing only Pearson residuals
- **glm-diagnostics.AC7.3 Success:** `eqx.filter_jit(glmax.check)(fitted)` compiles and produces the same results as the non-JIT call
- **glm-diagnostics.AC7.4 Failure:** `glmax.check("not_a_fitted_glm")` raises `TypeError`

---

## Key Codebase Facts

- Current `diagnostics.py` has: `Diagnostics` NamedTuple (placeholder, no fields), `check(fitted) -> Diagnostics` (raises `TypeError` for non-FittedGLM)
- Current `__init__.py:26`: `from .diagnostics import check as check, Diagnostics as Diagnostics`
- Current `__init__.py:57`: `"Diagnostics"` in `__all__`
- Existing `tests/diagnostics/test_check.py` tests reference `glmax.Diagnostics` — these must be rewritten
- `eqx.filter_jit` is already used throughout the codebase (verify: `_fit/irls.py` uses it)
- Diagnostic class instances have no array fields → treated as static in `filter_jit`
- `diagnostics` argument: a Python tuple of `AbstractDiagnostic` instances → static under `filter_jit` (non-array leaves)
- Run tests: `pytest -p no:capture tests`

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Replace `check()` stub and add `DEFAULT_DIAGNOSTICS` to `diagnostics.py`

**Verifies:** glm-diagnostics.AC7.1, glm-diagnostics.AC7.2, glm-diagnostics.AC7.3, glm-diagnostics.AC7.4

**Files:**
- Modify: `src/glmax/diagnostics.py`

**Implementation:**

In `diagnostics.py`:

1. Add `import equinox as eqx` to the top-level imports (it should already be there from Phase 2; verify).
2. Remove the old `Diagnostics` NamedTuple class entirely.
3. Replace the old `check()` function with the new implementation below.
4. Update `__all__` to remove `"Diagnostics"` and add `"DEFAULT_DIAGNOSTICS"`.

The final `__all__` for `diagnostics.py` should be:

```python
__all__ = [
    "AbstractDiagnostic",
    "DEFAULT_DIAGNOSTICS",
    "DevianceResidual",
    "GoodnessOfFit",
    "GofStats",
    "Influence",
    "InfluenceStats",
    "PearsonResidual",
    "QuantileResidual",
    "check",
]
```

Add after the `Influence` class:

```python
DEFAULT_DIAGNOSTICS: tuple[AbstractDiagnostic, ...] = (
    PearsonResidual(),
    DevianceResidual(),
    QuantileResidual(),
    GoodnessOfFit(),
    Influence(),
)


@eqx.filter_jit
def check(
    fitted: FittedGLM,
    diagnostics: tuple[AbstractDiagnostic, ...] = DEFAULT_DIAGNOSTICS,
) -> tuple:
    r"""Assess model fit and return a tuple of diagnostic results.

    The canonical `check` grammar verb. Accepts any tuple of
    `AbstractDiagnostic` instances and returns a positionally-matched tuple
    of results — each element `T` corresponding to `diagnostic.diagnose(fitted)`.

    Decorated with `eqx.filter_jit`; JIT-compiles on first call and caches
    subsequent calls with the same structure.

    **Arguments:**

    - `fitted`: `FittedGLM` noun produced by `fit(...)`.
    - `diagnostics`: tuple of `AbstractDiagnostic` instances to apply.
      Defaults to `DEFAULT_DIAGNOSTICS` which computes all five built-in
      diagnostics.

    **Returns:**

    Tuple of diagnostic results, one per entry in `diagnostics`, in the
    same positional order.

    **Raises:**

    - `TypeError`: if `fitted` is not a `FittedGLM` instance.
    """
    if not isinstance(fitted, FittedGLM):
        raise TypeError(
            f"check(...) expects `fitted` to be a FittedGLM instance, "
            f"got {type(fitted).__name__!r}."
        )
    return tuple(d.diagnose(fitted) for d in diagnostics)
```

**Note on JIT and isinstance guard:** `isinstance` is executed at Python trace time (before JAX traces arrays), so the `TypeError` will always be raised for the wrong type, even under JIT. This is consistent with how Equinox handles type guards.

**Verification:**

Run: `python -c "import glmax; print(glmax.DEFAULT_DIAGNOSTICS)"`
Expected: prints a 5-tuple of diagnostic instances without error

**Commit:** `feat(diagnostics): wire check() with eqx.filter_jit and DEFAULT_DIAGNOSTICS`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Update `__init__.py` exports

**Verifies:** glm-diagnostics.AC7.1 (public API surface)

**Files:**
- Modify: `src/glmax/__init__.py`

**Implementation:**

Replace the existing diagnostics import line (line 26):
```python
from .diagnostics import check as check, Diagnostics as Diagnostics
```

With:
```python
from .diagnostics import (
    AbstractDiagnostic as AbstractDiagnostic,
    check as check,
    DEFAULT_DIAGNOSTICS as DEFAULT_DIAGNOSTICS,
    DevianceResidual as DevianceResidual,
    GoodnessOfFit as GoodnessOfFit,
    GofStats as GofStats,
    Influence as Influence,
    InfluenceStats as InfluenceStats,
    PearsonResidual as PearsonResidual,
    QuantileResidual as QuantileResidual,
)
```

Replace `"Diagnostics"` in `__all__` with the new names:

```python
__all__ = [
    "GLMData",
    "Params",
    "GLM",
    "AbstractFitter",
    "FitResult",
    "FittedGLM",
    "InferenceResult",
    "AbstractDiagnostic",
    "DEFAULT_DIAGNOSTICS",
    "DevianceResidual",
    "GoodnessOfFit",
    "GofStats",
    "Influence",
    "InfluenceStats",
    "PearsonResidual",
    "QuantileResidual",
    "AbstractTest",
    "WaldTest",
    "ScoreTest",
    "AbstractStdErrEstimator",
    "FisherInfoError",
    "HuberError",
    "specify",
    "predict",
    "fit",
    "infer",
    "check",
]
```

**Verification:**

Run: `python -c "import glmax; print(glmax.PearsonResidual, glmax.DEFAULT_DIAGNOSTICS)"`
Expected: prints class and tuple without error

**Commit:** `feat: export new diagnostic classes from glmax public API`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Update `tests/diagnostics/test_check.py` and run full suite

**Verifies:** glm-diagnostics.AC7.1, glm-diagnostics.AC7.2, glm-diagnostics.AC7.3, glm-diagnostics.AC7.4

**Files:**
- Modify: `tests/diagnostics/test_check.py` (rewrite to new API)

**Implementation:**

Replace the entire contents of `tests/diagnostics/test_check.py` with:

```python
# pattern: Imperative Shell

import pytest

import equinox as eqx
import jax.numpy as jnp

import glmax
from glmax import GLMData
from glmax.diagnostics import (
    DEFAULT_DIAGNOSTICS,
    DevianceResidual,
    GoodnessOfFit,
    GofStats,
    Influence,
    InfluenceStats,
    PearsonResidual,
    QuantileResidual,
)
from glmax.family import Gaussian


def _make_fitted():
    model = glmax.specify(family=Gaussian())
    data = GLMData(
        X=jnp.array([[1.0], [2.0], [3.0], [4.0]]),
        y=jnp.array([1.2, 1.9, 3.1, 4.2]),
    )
    return glmax.fit(model, data)


def test_check_default_returns_5_tuple():
    fitted = _make_fitted()
    result = glmax.check(fitted)
    assert isinstance(result, tuple)
    assert len(result) == 5


def test_check_default_positional_types():
    fitted = _make_fitted()
    pearson, deviance, quantile, gof, influence = glmax.check(fitted)
    assert isinstance(pearson, jnp.ndarray)
    assert isinstance(deviance, jnp.ndarray)
    assert isinstance(quantile, jnp.ndarray)
    assert isinstance(gof, GofStats)
    assert isinstance(influence, InfluenceStats)


def test_check_custom_diagnostics_single():
    fitted = _make_fitted()
    result = glmax.check(fitted, diagnostics=(PearsonResidual(),))
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].shape == (4,)


def test_check_custom_diagnostics_two():
    fitted = _make_fitted()
    result = glmax.check(fitted, diagnostics=(PearsonResidual(), DevianceResidual()))
    assert len(result) == 2


def test_check_rejects_non_fitted_glm():
    with pytest.raises(TypeError, match="FittedGLM"):
        glmax.check("not_a_fitted_glm")


def test_check_filter_jit_produces_same_result():
    fitted = _make_fitted()
    result_eager = glmax.check(fitted, diagnostics=(PearsonResidual(),))
    result_jit = eqx.filter_jit(glmax.check)(fitted, diagnostics=(PearsonResidual(),))
    assert jnp.allclose(result_eager[0], result_jit[0], atol=1e-10)


def test_check_default_all_outputs_finite():
    fitted = _make_fitted()
    pearson, deviance, quantile, gof, influence = glmax.check(fitted)
    assert jnp.all(jnp.isfinite(pearson))
    assert jnp.all(jnp.isfinite(deviance))
    assert jnp.all(jnp.isfinite(quantile))
    for field in (gof.deviance, gof.aic, gof.bic, gof.df_resid, gof.pearson_chi2, gof.dispersion):
        assert jnp.isfinite(field)
    assert jnp.all(jnp.isfinite(influence.leverage))
    assert jnp.all(jnp.isfinite(influence.cooks_distance))
```

**Verification:**

Run: `pytest -p no:capture tests/diagnostics/test_check.py`
Expected: All 7 tests pass

Run: `pytest -p no:capture tests`
Expected: All tests pass (old Diagnostics-based tests gone, new API tests in place)

**Commit:** `feat(diagnostics): update test_check.py to new check() tuple API`
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->
