# Inferrer Abstraction Implementation Plan — Phase 3

**Goal:** Wire `infer()` to accept and delegate to `AbstractInferrer`, update the shell, expand the public surface with six new exports, and add tests for the new delegation ACs.

**Architecture:** `inference.py`'s `infer()` becomes a thin boundary that lazily imports `AbstractInferrer`/`DEFAULT_INFERRER` inside its body (avoiding circular imports), validates all three arguments, then delegates to `inferrer(fitted, stderr)`. The imperative shell threads the new parameter through. `glmax/__init__.py` gains six new import-and-export lines. Tests confirm delegation routing, TypeError gates, and import surface.

**Tech Stack:** Python, JAX, Equinox, pytest

**Scope:** Phase 3 of 3 (wire infer and expand public surface)

**Codebase verified:** 2026-03-16

---

## Acceptance Criteria Coverage

### inferrer-abstraction.AC4: `infer()` signature and delegation
- **inferrer-abstraction.AC4.1 Success:** `infer(fitted)` with no extra args produces same result as pre-refactor
- **inferrer-abstraction.AC4.2 Success:** `infer(fitted, inferrer=ScoreInferrer())` routes to `ScoreInferrer`
- **inferrer-abstraction.AC4.3 Success:** `infer(fitted, stderr=HuberError())` passes `HuberError` into `WaldInferrer`
- **inferrer-abstraction.AC4.4 Failure:** `infer(fitted, inferrer=object())` raises `TypeError`
- **inferrer-abstraction.AC4.5 Failure:** `infer(fitted, stderr=object())` raises `TypeError`

### inferrer-abstraction.AC5: Public surface exports
- **inferrer-abstraction.AC5.1 Success:** `from glmax import AbstractInferrer, WaldInferrer, ScoreInferrer` succeeds
- **inferrer-abstraction.AC5.2 Success:** `from glmax import AbstractStdErrEstimator, FisherInfoError, HuberError` succeeds
- **inferrer-abstraction.AC5.3 Success:** all six names appear in `glmax.__all__`

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Update `infer()` in `inference.py` to delegate to the inferrer

**Verifies:** inferrer-abstraction.AC4.1, inferrer-abstraction.AC4.4, inferrer-abstraction.AC4.5

**Files:**
- Modify: `src/glmax/infer/inference.py`

**Implementation:**

Replace the `infer()` function body so it lazily imports `AbstractInferrer`/`DEFAULT_INFERRER` from `.inferrer`, validates all four boundary conditions, then delegates entirely to `inferrer(fitted, stderr)`. The existing Wald computation (lines 69–82 after Phase 1 rename) is now in `WaldInferrer.__call__` — remove it from `inference.py`.

Keep `_matches_fit_result_shape` in the import and in `infer()`'s body. The `FitResult`-shape guard must stay in `infer()` to preserve the existing `TypeError("FitResult")` contract tested by `test_infer_rejects_invalid_model_and_result_contracts`. Without it, a `result=object()` argument would reach `WaldInferrer` and raise `AttributeError` on attribute access instead of the documented `TypeError`.

Remove only the now-unused `validate_fit_result` from `inference.py`'s `..fit` import line — it remains needed in `inferrer.py` where both `WaldInferrer.__call__` and `ScoreInferrer.__call__` call it. Do not remove it from `inferrer.py`.

The resulting import in `inference.py` must retain all four of: `_matches_fit_result_shape` (used in the FitResult guard), `_matches_fitted_glm_shape` (used in the FittedGLM guard), `FittedGLM` (used in the function signature), `Params` (used in `InferenceResult` field annotation).

Because `ScoreInferrer` also calls `validate_fit_result`, the existing `test_infer_rejects_invalid_fit_artifacts_deterministically` test cases (NaN betas, wrong dtypes, non-finite disp) continue to pass when using either inferrer — `infer()` runs the `_matches_fit_result_shape` type guard first, and the inferrer runs the deeper content check.

The complete updated file:

```python
# pattern: Functional Core

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp

from jax import Array
from jax.scipy.stats import norm
from jaxtyping import ArrayLike

from ..family.dist import ExponentialFamily, Gaussian
from ..family.utils import t_cdf
from ..fit import _matches_fit_result_shape, _matches_fitted_glm_shape, FittedGLM, Params
from .stderr import AbstractStdErrEstimator, FisherInfoError


if TYPE_CHECKING:
    pass


__all__ = ["InferenceResult", "infer", "wald_test"]


DEFAULT_STDERR: AbstractStdErrEstimator = FisherInfoError()


class InferenceResult(NamedTuple):
    """Canonical _infer verb output contract."""

    params: Params
    se: Array
    stat: Array
    p: Array


def wald_test(statistic: ArrayLike, df: int, family: ExponentialFamily) -> Array:
    r"""Two-sided Wald test p-values.

    Uses a $t_{df}$ distribution for Gaussian families and $\mathcal{N}(0, 1)$
    for all others.

    **Arguments:**

    - `statistic`: test statistics $\hat\beta / \mathrm{SE}(\hat\beta)$, shape `(p,)`.
    - `df`: residual degrees of freedom $n - p$.
    - `family`: fitted `ExponentialFamily` instance.

    **Returns:**

    Two-sided p-values, shape `(p,)`.
    """
    if isinstance(family, Gaussian):
        return 2 * t_cdf(-jnp.abs(statistic), df)
    return 2 * norm.sf(jnp.abs(statistic))


def infer(
    fitted: FittedGLM,
    inferrer=None,
    stderr: AbstractStdErrEstimator = DEFAULT_STDERR,
) -> InferenceResult:
    """Inferential summaries from fit artifacts without refitting."""
    from .inferrer import AbstractInferrer as _AbstractInferrer, DEFAULT_INFERRER as _DEFAULT_INFERRER

    if inferrer is None:
        inferrer = _DEFAULT_INFERRER

    if not _matches_fitted_glm_shape(fitted):
        raise TypeError("_infer(...) expects `fitted` to be a FittedGLM instance.")
    if not _matches_fit_result_shape(fitted.result):
        raise TypeError("_infer(...) expects `fitted.result` to be a FitResult instance.")
    if not isinstance(inferrer, _AbstractInferrer):
        raise TypeError("_infer(...) expects `inferrer` to be an AbstractInferrer instance.")
    if not isinstance(stderr, AbstractStdErrEstimator):
        raise TypeError("_infer(...) expects `stderr` to be an AbstractStdErrEstimator instance.")

    return inferrer(fitted, stderr)
```

**Verification:**

Run: `pytest -p no:capture tests`
Expected: FAILURES — shell still expects old signature, and `test_infer_signature_matches_canonical_surface` will fail. This is expected before Task 2.

**Commit:** Do NOT commit yet — complete remaining tasks first.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Update `infer/__init__.py` shell to thread `inferrer`

**Verifies:** inferrer-abstraction.AC4.1, inferrer-abstraction.AC4.2, inferrer-abstraction.AC4.3, inferrer-abstraction.AC4.4, inferrer-abstraction.AC4.5

**Files:**
- Modify: `src/glmax/infer/__init__.py`

**Implementation:**

Add `inferrer=None` as a positional-or-keyword parameter between `fitted` and `stderr`. Use a `kwargs`-build pattern so only non-`None` arguments are forwarded to `_infer`, preserving the lazy-default behaviour in `inference.py`.

Add `AbstractInferrer` to the `TYPE_CHECKING` block for type-checker support only — this is a static-analysis annotation, not a runtime re-export. Do NOT add runtime re-exports of inferrer types to this module (see note below).

The complete updated file:

```python
# pattern: Imperative Shell

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ..fit import FittedGLM
    from .diagnostics import Diagnostics
    from .inference import InferenceResult
    from .inferrer import AbstractInferrer
    from .stderr import AbstractStdErrEstimator


def infer(
    fitted: "FittedGLM",
    inferrer: "AbstractInferrer" = None,
    stderr: "AbstractStdErrEstimator" = None,
) -> "InferenceResult":
    """Canonical _infer verb entrypoint."""
    from .inference import infer as _infer

    kwargs = {}
    if inferrer is not None:
        kwargs["inferrer"] = inferrer
    if stderr is not None:
        kwargs["stderr"] = stderr
    return _infer(fitted, **kwargs)


def check(fitted: "FittedGLM") -> "Diagnostics":
    """Canonical check verb entrypoint."""
    from .diagnostics import check as _check

    return _check(fitted)


__all__ = ["infer", "check"]
```

**Note on re-exports:** Do NOT re-export `AbstractInferrer`, `WaldInferrer`, `ScoreInferrer`, or `DEFAULT_INFERRER` from `glmax.infer/__init__.py`. The design says these are accessible from top-level `glmax.*` (AC5), and `glmax.infer` already withholds `FisherInfoError`/`HuberError` from its namespace (enforced by `test_infer_shims_are_not_publicly_reexported`). Adding inferrer types to `glmax.infer` but not SE types would create an unexplained asymmetry. The primary surface is `glmax.*`; advanced users who need the internals can import directly from `glmax.infer.inferrer`.

**Verification:**

Run: `pytest -p no:capture tests`
Expected: still some failures (signature test, exports test). Continue to Task 3.

**Commit:** Do NOT commit yet.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Expand public surface in `glmax/__init__.py`

**Verifies:** inferrer-abstraction.AC5.1, inferrer-abstraction.AC5.2, inferrer-abstraction.AC5.3

**Files:**
- Modify: `src/glmax/__init__.py`

**Implementation:**

Add six new import-and-export lines. Three come from `.infer.inferrer`, three from `.infer.stderr`. Add all six to `__all__` (total 19 items).

The complete updated file:

```python
# pattern: Imperative Shell

from importlib.metadata import version  # pragma: no cover

import jax

from .data import GLMData as GLMData
from .fit import (
    fit as fit,
    FitResult as FitResult,
    FittedGLM as FittedGLM,
    Fitter as Fitter,
    Params as Params,
    predict as predict,
)
from .glm import GLM as GLM, specify as specify
from .infer import check as check, infer as infer
from .infer.diagnostics import Diagnostics as Diagnostics
from .infer.inference import InferenceResult as InferenceResult
from .infer.inferrer import (
    AbstractInferrer as AbstractInferrer,
    WaldInferrer as WaldInferrer,
    ScoreInferrer as ScoreInferrer,
)
from .infer.stderr import (
    AbstractStdErrEstimator as AbstractStdErrEstimator,
    FisherInfoError as FisherInfoError,
    HuberError as HuberError,
)


jax.config.update("jax_enable_x64", True)  # noqa: E402

__version__ = version("glmax")

__all__ = [
    "GLMData",
    "Params",
    "GLM",
    "Fitter",
    "FitResult",
    "FittedGLM",
    "InferenceResult",
    "Diagnostics",
    "AbstractInferrer",
    "WaldInferrer",
    "ScoreInferrer",
    "AbstractStdErrEstimator",
    "FisherInfoError",
    "HuberError",
    "specify",
    "predict",
    "fit",
    "_infer",
    "check",
]
```

**Verification:**

Run: `pytest -p no:capture tests`
Expected: still some failures (test_fit_api.py assertions about __all__ count and infer signature). Continue to Task 4.

**Commit:** Do NOT commit yet.
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-5) -->

<!-- START_TASK_4 -->
### Task 4: Update `test_fit_api.py` for new signature and expanded exports

**Verifies:** inferrer-abstraction.AC4.1, inferrer-abstraction.AC5.1, inferrer-abstraction.AC5.2, inferrer-abstraction.AC5.3

**Files:**
- Modify: `tests/test_fit_api.py`

**Implementation:**

Two tests require updates.

**1. `test_top_level_exports_are_canonical_nouns_and_verbs` (line 37–52)**

Add the six new names to the expected set:

```python
def test_top_level_exports_are_canonical_nouns_and_verbs() -> None:
    assert set(glmax.__all__) == {
        "GLMData",
        "Params",
        "GLM",
        "Fitter",
        "FitResult",
        "FittedGLM",
        "InferenceResult",
        "Diagnostics",
        "AbstractInferrer",
        "WaldInferrer",
        "ScoreInferrer",
        "AbstractStdErrEstimator",
        "FisherInfoError",
        "HuberError",
        "specify",
        "predict",
        "fit",
        "_infer",
        "check",
    }
```

**2. `test_infer_signature_matches_canonical_surface` (line 137–141)**

Add `inferrer` to the parameter list and assert its kind:

```python
def test_infer_signature_matches_canonical_surface() -> None:
    sig = inspect.signature(glmax.infer)
    assert list(sig.parameters) == ["fitted", "inferrer", "stderr"]
    assert sig.parameters["fitted"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["inferrer"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["stderr"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    # inferrer default must be None (lazy-resolved inside _infer() body, not at module load)
    assert sig.parameters["inferrer"].default is None
```

**Note on `test_infer_shims_are_not_publicly_reexported`:** This test (lines 89–102) continues to pass unchanged. It checks that `FisherInfoError` and `HuberError` are NOT attributes of `glmax.infer` — they remain accessible only via `glmax.*`. Task 2 does NOT add inferrer types as runtime attributes of `glmax.infer` (only a `TYPE_CHECKING` import), so the test is unaffected.

**Note on AC4.5 coverage:** AC4.5 (`infer(fitted, stderr=object())` raises `TypeError`) is already covered by the existing `test_infer_rejects_invalid_model_and_result_contracts` assertion at line 82–83 of `test_infer_verbs.py`. No new test is needed for AC4.5 — confirm it still passes after the signature change.

**Note on `params` identity:** `test_infer_returns_inference_result_without_refitting` (line 54) asserts `inferred.params is fitted.params` using object identity. This is preserved because `WaldInferrer` returns `InferenceResult(params=fit_result.params, ...)` where `fit_result.params` is the same `Params` object held by `FittedGLM`. No code change is needed, but be aware the test relies on identity, not equality.

**Verification:**

Run: `pytest -p no:capture tests`
Expected: still failures in `test_infer_verbs.py` for the new delegation tests (Task 5 not done yet). Continue.

**Commit:** Do NOT commit yet.
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Add delegation and surface tests in `test_infer_verbs.py`

**Verifies:** inferrer-abstraction.AC4.1, inferrer-abstraction.AC4.2, inferrer-abstraction.AC4.3, inferrer-abstraction.AC4.4, inferrer-abstraction.AC5.1, inferrer-abstraction.AC5.2

**Files:**
- Modify: `tests/test_infer_verbs.py`

**Implementation:**

Add these tests at the end of the file. Start by updating the import block at the top of the file — add the new types:

```python
from glmax import FittedGLM, GLMData, InferenceResult, Params
from glmax.family import Gaussian
from glmax._infer.hyptest import ScoreTest, WaldTest
from glmax._infer.stderr import AbstractStdErrEstimator, HuberError
```

Then add the following test functions.

**AC4.1 — default inferrer produces same results as WaldInferrer explicit:**

```python
def test_infer_default_inferrer_matches_explicit_wald_inferrer() -> None:
    fitted = _make_fitted()

    result_default = glmax.infer(fitted)
    result_explicit = glmax.infer(fitted, inferrer=WaldInferrer())

    assert jnp.allclose(result_default.stat, result_explicit.stat)
    assert jnp.allclose(result_default.se, result_explicit.se)
    assert jnp.allclose(result_default.p, result_explicit.p)
```

**AC4.2 — ScoreInferrer route (se all-NaN, stat and p valid):**

```python
def test_infer_routes_to_score_inferrer_when_specified() -> None:
    fitted = _make_fitted()

    result = glmax.infer(fitted, inferrer=ScoreInferrer())

    assert isinstance(result, InferenceResult)
    assert jnp.all(jnp.isnan(result.se))
    assert jnp.all(jnp.isfinite(result.stat))
    assert jnp.all((result.p >= 0.0) & (result.p <= 1.0))
    assert result.stat.shape == fitted.params.beta.shape
```

**AC4.3 — custom stderr reaches WaldInferrer (via recording stub):**

Note: `AbstractStdErrEstimator` is `eqx.Module, strict=True`. Define stubs **without** `strict=True` and do **not** declare any new instance attributes on the stub class — use closed-over mutable values (e.g., dicts) for state instead. This is the same pattern used by `RecordingStdErr` in the existing `test_infer_uses_injected_stderr_estimator` test.

```python
def test_infer_passes_stderr_into_wald_inferrer() -> None:
    fitted = _make_fitted()
    call_count = {"n": 0}

    class CountingStdErr(AbstractStdErrEstimator):  # no strict=True; no instance attrs
        def __call__(self, fitted_arg):
            call_count["n"] += 1
            # return a valid 1×1 covariance matrix (fitted uses 1-column X)
            return jnp.array([[4.0]])

    result = glmax.infer(fitted, stderr=CountingStdErr())

    assert call_count["n"] == 1, "WaldInferrer must call stderr exactly once"
    # Verify the returned covariance was actually used.
    # _make_fitted() in test_infer_verbs.py uses 1-column X, so beta is shape (1,).
    # The (1,1) covariance [[4.0]] gives se = sqrt(diag([[4.0]])) = [2.0].
    assert result.se.shape == fitted.params.beta.shape
    assert jnp.allclose(result.se, jnp.array([2.0]))
```

**AC4.4 — TypeError for wrong-type inferrer:**

Add a new assertion to the existing `test_infer_rejects_invalid_model_and_result_contracts`. Find the function and add at the end:

```python
    with pytest.raises(TypeError, match="AbstractInferrer"):
        glmax.infer(fitted, inferrer=object())
```

**AC5.1 and AC5.2 — import surface:**

```python
def test_inferrer_types_are_importable_from_top_level_glmax() -> None:
    from glmax import AbstractTest, WaldTest, ScoreTest  # noqa: F401

    assert AbstractTest is not None
    assert WaldTest is not None
    assert ScoreTest is not None


def test_stderr_types_are_importable_from_top_level_glmax() -> None:
    from glmax import AbstractStdErrEstimator, FisherInfoError, HuberError  # noqa: F401

    assert AbstractStdErrEstimator is not None
    assert FisherInfoError is not None
    assert HuberError is not None
```

**Verification:**

Run: `pytest -p no:capture tests`
Expected: `217 passed` (baseline) + new tests → total passes increase; **zero failures**.

**Commit:**

```bash
git add src/glmax/_infer/infer.py \
        src/glmax/_infer/__init__.py \
        src/glmax/__init__.py \
        tests/test_fit_api.py \
        tests/test_infer_verbs.py
git commit -m "feat: wire inferrer argument into infer() and expand public surface

infer() now accepts an AbstractInferrer argument (default WaldInferrer).
The infer shell and core both updated. Six new names exported from glmax:
AbstractInferrer, WaldInferrer, ScoreInferrer, AbstractStdErrEstimator,
FisherInfoError, HuberError.

inferrer-abstraction.AC4.1, AC4.2, AC4.3, AC4.4, AC4.5
inferrer-abstraction.AC5.1, AC5.2, AC5.3"
```
<!-- END_TASK_5 -->

<!-- END_SUBCOMPONENT_B -->

## Review Notes

### Phase 3 Red/Green Evidence

- Red before implementation:
  - `PYTHONPATH=src pytest -p no:capture /tmp/glmax_phase3_subcomponent_a_tmp_test.py`
  - Result: `1 failed`
  - Relevant failure: `ImportError: cannot import name 'AbstractInferrer' from 'glmax'`
- Red before stale test updates:
  - `pytest -p no:capture tests/test_fit_api.py -k 'top_level_exports_are_canonical_nouns_and_verbs or infer_signature_matches_canonical_surface'`
  - Result: failed because `glmax.__all__` had six new exports and `glmax.infer` had the extra `inferrer` parameter.
- Green after implementation:
  - `PYTHONPATH=src pytest -p no:capture /tmp/glmax_phase3_subcomponent_a_tmp_test.py`
  - Result: `1 passed`
- Green after test updates:
  - `pytest -p no:capture tests/test_fit_api.py tests/test_infer_verbs.py`
  - Result: `46 passed`
