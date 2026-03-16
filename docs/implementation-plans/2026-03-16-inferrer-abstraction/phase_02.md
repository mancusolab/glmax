# Inferrer Abstraction Implementation Plan — Phase 2

**Goal:** Create `src/glmax/infer/inferrer.py` containing `AbstractInferrer`, `WaldInferrer`,
`ScoreInferrer`, and `DEFAULT_INFERRER`. `WaldInferrer` must produce results numerically identical
to the pre-refactor `infer()`. `ScoreInferrer` must produce valid per-coefficient score statistics
without calling `stderr`.

**Architecture:** New module `inferrer.py` inside `src/glmax/infer/`, mirroring the `stderr.py`
pattern (abstract eqx.Module + concrete subclasses). `WaldInferrer` migrates the current Wald
logic. `ScoreInferrer` computes score statistics directly from fit artifacts. Module-level imports
from `inference.py` are safe in Phase 2. Phase 3 will introduce a circular import
(`inference.py` will import from `inferrer.py`, which already imports `InferenceResult`/`wald_test`
from `inference.py`). Phase 3 resolves this immediately with a lazy import inside `infer()`'s
body; no Phase 2 action is needed.

**Tech Stack:** Python, JAX (`jnp`, `jax.scipy.stats.norm`), Equinox, pytest

**Scope:** Phase 2 of 3 (new `inferrer.py` module + tests; does NOT modify `infer()` signature yet)

**Codebase verified:** 2026-03-16

---

## Acceptance Criteria Coverage

### inferrer-abstraction.AC2: `WaldInferrer` correctness
- **inferrer-abstraction.AC2.1 Success:** `WaldInferrer()(fitted, stderr)` returns `InferenceResult` with same p-values as pre-refactor `infer()` to float64 precision
- **inferrer-abstraction.AC2.2 Success:** Gaussian family uses t-distribution; all other families use standard normal
- **inferrer-abstraction.AC2.3 Success:** `WaldInferrer` calls `stderr(fitted)` internally and uses the resulting covariance to compute SE
- **inferrer-abstraction.AC2.4 Failure:** non-`FittedGLM` first arg raises `TypeError`

### inferrer-abstraction.AC3: `ScoreInferrer` correctness
- **inferrer-abstraction.AC3.1 Success:** for regular fitted models with positive finite scale/information, `ScoreInferrer()(fitted, stderr)` returns `InferenceResult` with `stat` finite, `p` in `[0,1]`, `se` all-NaN
- **inferrer-abstraction.AC3.2 Success:** `ScoreInferrer` does not call `stderr` (verified via a raising stderr stub)
- **inferrer-abstraction.AC3.3 Success:** `stat` shape matches `(p,)` for all supported families
- **inferrer-abstraction.AC3.4 Edge:** for Gaussian, `ScoreInferrer` matches the direct MLE-point score-style formula `Xᵀ (glm_wt * score_residual) / sqrt(phi * diag(Xᵀ diag(glm_wt) X))` and produces valid two-sided p-values
- **inferrer-abstraction.AC3.5 Failure:** degenerate scale or Fisher information raises `ValueError` instead of returning invalid statistics

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->

<!-- START_TASK_1 -->
### Task 1: Write failing tests for WaldInferrer

**Verifies:** inferrer-abstraction.AC2.1, inferrer-abstraction.AC2.2, inferrer-abstraction.AC2.3, inferrer-abstraction.AC2.4

**Files:**
- Create: `tests/test_inferrers.py`

**Implementation:**

Create `tests/test_inferrers.py`. This file tests `WaldInferrer` and `ScoreInferrer` directly
(as standalone strategy objects), mirroring how `tests/test_fitters.py` tests `FisherInfoError`
and `HuberError`.

Imports to use at the top of the file:

```python
import jax.numpy as jnp
import equinox as eqx
import pytest

import glmax
from glmax._infer import InferenceResult
from glmax._infer.infer import infer as legacy_infer
# Internal imports: testing strategy objects directly, not the public glmax.* surface
from glmax._infer.hyptest import WaldTest, ScoreTest, AbstractTest
from glmax._infer.stderr import AbstractStdErrEstimator, FisherInfoError
from glmax.family import Gaussian, Poisson, Binomial, NegativeBinomial
from glmax import GLMData
```

Helper to build a fitted model (mirrors `_make_fitted()` in `test_infer_verbs.py`):

```python
def _make_fitted(family=None):
    if family is None:
        family = Gaussian()
    model = glmax.specify(family=family)
    data = GLMData(
        X=jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
        y=jnp.array([1.2, 1.9, 3.1, 4.2]),
    )
    return glmax.fit(model, data)
```

Tests to write for WaldInferrer:

- `test_wald_inferrer_matches_legacy_infer`: Call `legacy_infer(fitted)` and
  `WaldInferrer()(fitted, FisherInfoError())`. Assert `stat`, `se`, `p` match to float64 precision
  using `jnp.allclose(atol=1e-12)`. Verifies AC2.1.

- `test_wald_inferrer_gaussian_uses_t_distribution`: For Gaussian, p-values should be computed
  via t-CDF. Pass a stat array to both the legacy `wald_test` and via `WaldInferrer`, confirm
  they agree. (Indirect verification via regression against legacy.) Verifies AC2.2.

- `test_wald_inferrer_uses_injected_stderr`: Create a `ConstantCovStdErr(AbstractStdErrEstimator)`
  that returns a known 2×2 covariance (e.g., `jnp.eye(2) * 4.0`). Call
  `WaldInferrer()(fitted, ConstantCovStdErr())`. Assert `se` equals `jnp.array([2.0, 2.0])`
  (sqrt of diagonal). This confirms stderr was called and its result used. Verifies AC2.3.

- `test_wald_inferrer_rejects_non_fitted_glm`: Call `WaldInferrer()(object(), FisherInfoError())`.
  Assert raises `TypeError`. Verifies AC2.4.

**Testing:**
Tests must verify each AC listed above. Follow the pattern in `tests/test_fitters.py` for
`FisherInfoError` / `HuberError` tests. Use `AbstractStdErrEstimator` subclasses as stubs where
needed — subclass without `strict=True` and override `__call__` only.

**Verification:**

Run: `pytest -p no:capture tests/test_inferrers.py`
Expected: `ImportError` or collection errors — `inferrer.py` does not exist yet.

**Commit:** Do NOT commit yet.
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Create `inferrer.py` with `AbstractInferrer` and `WaldInferrer`

**Verifies:** inferrer-abstraction.AC2.1, inferrer-abstraction.AC2.2, inferrer-abstraction.AC2.3, inferrer-abstraction.AC2.4

**Files:**
- Create: `src/glmax/infer/inferrer.py`

**Implementation:**

Create `src/glmax/infer/inferrer.py` with the following content:

```python
# pattern: Functional Core

"""Inferrer strategies for the `_infer()` verb."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.scipy.stats import norm

from ..family.dist import ExponentialFamily, Gaussian
from ..family.utils import t_cdf
from ..fit import _matches_fitted_glm_shape, validate_fit_result
from .inference import InferenceResult, wald_test
from .stderr import AbstractStdErrEstimator

if TYPE_CHECKING:
    from ..fit import FittedGLM


__all__ = ["AbstractInferrer", "WaldInferrer", "ScoreInferrer", "DEFAULT_INFERRER"]


class AbstractInferrer(eqx.Module, strict=True):
    """Base class for inference strategies used by `_infer(fitted, inferrer=...)`."""

    @abstractmethod
    def __call__(
        self,
        fitted: "FittedGLM",
        stderr: AbstractStdErrEstimator,
    ) -> InferenceResult:
        """Compute inferential summaries from a fitted GLM.

        **Arguments:**

        - `fitted`: validated `FittedGLM` from `fit()`.
        - `stderr`: standard-error estimator; concrete inferrers call it only if needed.

        **Returns:**

        `InferenceResult` with `(params, se, stat, p)`.
        """


class WaldInferrer(AbstractInferrer, strict=True):
    """Wald (z/t) hypothesis test. Default inferrer for `_infer()`.

    Calls `stderr(fitted)` to compute the covariance matrix, extracts per-coefficient
    standard errors, and computes two-sided p-values via the Wald statistic
    `stat_j = β̂_j / SE(β̂_j)`. Uses a t-distribution for Gaussian families and
    the standard normal for all others.
    """

    def __call__(
        self,
        fitted: "FittedGLM",
        stderr: AbstractStdErrEstimator,
    ) -> InferenceResult:
        if not _matches_fitted_glm_shape(fitted):
            raise TypeError("WaldInferrer expects `fitted` to be a FittedGLM instance.")
        if not isinstance(stderr, AbstractStdErrEstimator):
            raise TypeError(
                "WaldInferrer expects `stderr` to be an AbstractStdErrEstimator instance."
            )

        model = fitted.model
        fit_result = fitted.result
        validate_fit_result(fit_result)

        beta = jnp.asarray(fit_result.params.beta)
        covariance = jnp.asarray(stderr(fitted))
        se = jnp.sqrt(jnp.diag(covariance))
        stat = beta / se
        df = int(fit_result.eta.shape[0] - beta.shape[0])
        p = wald_test(stat, df, model.family)

        return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)
```

**Verification:**

Run: `pytest -p no:capture tests/test_inferrers.py -k "wald"`
Expected: WaldInferrer tests pass.

**Commit:** Do NOT commit yet — ScoreInferrer tests and implementation come next.
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Run WaldInferrer tests

**Verifies:** inferrer-abstraction.AC2.1, inferrer-abstraction.AC2.2, inferrer-abstraction.AC2.3, inferrer-abstraction.AC2.4

**Files:** None (verification only)

**Verification:**

Run: `pytest -p no:capture tests/test_inferrers.py -k "wald"`
Expected: All WaldInferrer tests pass.

Run: `pytest -p no:capture tests`
Expected: All 217+ tests pass (existing suite must stay green).

**Commit:** Do NOT commit yet.
<!-- END_TASK_3 -->

<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-6) -->

<!-- START_TASK_4 -->
### Task 4: Write failing tests for ScoreInferrer

**Verifies:** inferrer-abstraction.AC3.1, inferrer-abstraction.AC3.2, inferrer-abstraction.AC3.3, inferrer-abstraction.AC3.4

**Files:**
- Modify: `tests/test_inferrers.py`

**Implementation:**

Add ScoreInferrer tests to `tests/test_inferrers.py`.

Tests to write:

- `test_score_inferrer_returns_valid_result`: Call `ScoreInferrer()(fitted, FisherInfoError())`
  on a Gaussian fitted model. Assert:
  - `isinstance(result, InferenceResult)`
  - `bool(jnp.all(jnp.isfinite(result.stat)))` — stat is finite
  - `bool(jnp.all((result.p >= 0.0) & (result.p <= 1.0)))` — p in [0,1]
  - `bool(jnp.all(jnp.isnan(result.se)))` — se is all-NaN
  Verifies AC3.1.

- `test_score_inferrer_does_not_call_stderr`: Create `RaisingStdErr(AbstractStdErrEstimator)`
  that raises `RuntimeError("stderr should not be called")` in `__call__`. Call
  `ScoreInferrer()(fitted, RaisingStdErr())`. Assert no exception is raised. Verifies AC3.2.

- `test_score_inferrer_stat_shape_matches_beta`: For each of
  `[Gaussian(), Poisson(), Binomial()]`, assert `result.stat.shape == fitted.params.beta.shape`.
  Verifies AC3.3.

  Note: `NegativeBinomial` is intentionally excluded from this parametrize. `NB.scale()` internally
  calls `self.fit`, a known fragile self-recursion (see MEMORY.md known issues). This risk is
  explicitly accepted in design doc risk R3 for the NB family. The NB case is left untested here
  to avoid masking the deeper recursion issue.

  Note: `Binomial` requires integer y. Use a separate helper for Binomial:
  ```python
  def _make_binomial_fitted():
      model = glmax.specify(family=Binomial())
      data = GLMData(
          X=jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]),
          y=jnp.array([0.0, 1.0, 1.0, 0.0]),
      )
      return glmax.fit(model, data)
  ```

- `test_score_inferrer_gaussian_p_values_are_valid`: For Gaussian, call ScoreInferrer, assert
  p-values are in (0, 1] and finite. Also verify that the score statistic and p-values match the
  direct MLE-point score-style formula computed from `score_residual`, `glm_wt`, `phi`, and the
  Fisher-information diagonal. Verifies AC3.4.

- `test_score_inferrer_rejects_degenerate_gaussian_scale`: Build a perfect-fit Gaussian model
  where `family.scale(X, y, mu)` collapses to zero. Assert `ScoreInferrer()(fitted, stderr)`
  raises `ValueError` instead of returning NaN statistics or p-values. Verifies AC3.5.

**Verification:**

Run: `pytest -p no:capture tests/test_inferrers.py -k "score"`
Expected: `ImportError` — `ScoreInferrer` does not exist in `inferrer.py` yet.

**Commit:** Do NOT commit yet.
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Add `ScoreInferrer` and `DEFAULT_INFERRER` to `inferrer.py`

**Verifies:** inferrer-abstraction.AC3.1, inferrer-abstraction.AC3.2, inferrer-abstraction.AC3.3, inferrer-abstraction.AC3.4, inferrer-abstraction.AC3.5

**Files:**
- Modify: `src/glmax/infer/inferrer.py`

**Implementation:**

Append `ScoreInferrer` and `DEFAULT_INFERRER` to `src/glmax/infer/inferrer.py` after `WaldInferrer`:

```python
class ScoreInferrer(AbstractInferrer, strict=True):
    """Per-coefficient MLE-point score-style statistic. Does not call `stderr`.

    Computes the score vector `U_j = [Xᵀ diag(glm_wt) score_residual]_j / φ` and
    normalises by the square root of the Fisher information diagonal
    `I_jj = diag(Xᵀ diag(glm_wt) X)_j / φ`, giving
    `stat_j = U_j / √I_jj ~ N(0,1)` under the MLE-point approximation.

    `se` is set to NaN because no standard error is computed. Callers
    relying on `InferenceResult.se` downstream must handle NaN when using
    this inferrer.
    """

    def __call__(
        self,
        fitted: "FittedGLM",
        stderr: AbstractStdErrEstimator,  # intentionally ignored — ScoreInferrer does not call stderr
        # stderr type is not validated here because this inferrer never uses it;
        # _infer() at the boundary already guards isinstance(stderr, AbstractStdErrEstimator).
    ) -> InferenceResult:
        if not _matches_fitted_glm_shape(fitted):
            raise TypeError("ScoreInferrer expects `fitted` to be a FittedGLM instance.")

        fit_result = fitted.result
        validate_fit_result(fit_result)

        # All fit_result fields below are validated finite by validate_fit_result above
        X = fit_result.X
        y = fit_result.y
        mu = fit_result.mu
        glm_wt = fit_result.glm_wt
        score_residual = fit_result.score_residual
        beta = jnp.asarray(fit_result.params.beta)
        phi = jnp.asarray(fitted.model.family.scale(X, y, mu))

        # Numerator: X^T (glm_wt * score_residual), shape (p,)
        numerator = X.T @ (glm_wt * score_residual)

        # Fisher information diagonal: diag(X^T diag(glm_wt) X), shape (p,)
        fisher_diag = jnp.sum(X * (glm_wt[:, jnp.newaxis] * X), axis=0)

        if not bool(jnp.isfinite(phi)) or float(phi) <= 0.0:
            raise ValueError(
                "ScoreInferrer requires family.scale(X, y, mu) to be finite and > 0."
            )
        if not bool(jnp.all(jnp.isfinite(fisher_diag))) or not bool(jnp.all(fisher_diag > 0.0)):
            raise ValueError(
                "ScoreInferrer requires the Fisher information diagonal to be finite and > 0."
            )

        # stat_j = numerator_j / sqrt(phi * fisher_diag_j)
        stat = numerator / jnp.sqrt(phi * fisher_diag)

        # Two-sided p-values using standard normal
        p = 2.0 * norm.sf(jnp.abs(stat))

        # SE is not defined for the score test
        se = jnp.full(beta.shape, jnp.nan)

        return InferenceResult(params=fit_result.params, se=se, stat=stat, p=p)


DEFAULT_INFERRER: AbstractInferrer = WaldInferrer()
```

**Verification:**

Run: `pytest -p no:capture tests/test_inferrers.py`
Expected: All WaldInferrer and ScoreInferrer tests pass.

**Commit:** Do NOT commit yet.
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Run full test suite and commit

**Verifies:** inferrer-abstraction.AC2.1–AC2.4, inferrer-abstraction.AC3.1–AC3.5

**Files:** None (verification + commit)

**Verification:**

Run: `pytest -p no:capture tests`
Expected: All tests pass (217+ previously; new tests in test_inferrers.py increase count).

**Commit:**

```bash
git add src/glmax/_infer/hyptest.py tests/test_inferrers.py
git commit -m "feat: add AbstractInferrer, WaldInferrer, ScoreInferrer

New src/glmax/infer/inferrer.py introduces the AbstractInferrer
eqx.Module base class and two concrete implementations:

- WaldInferrer: migrates current Wald z/t test logic; calls
  stderr(fitted) internally to compute covariance and SE.
- ScoreInferrer: per-coefficient score (Rao) test computed directly
  from fit artifacts; ignores stderr; sets se=NaN.
- DEFAULT_INFERRER = WaldInferrer()

WaldInferrer output matches pre-refactor infer() to float64 precision.
ScoreInferrer stat is normalised by sqrt(phi * diag(X^T W X)).

inferrer-abstraction.AC2.1, AC2.2, AC2.3, AC2.4
inferrer-abstraction.AC3.1, AC3.2, AC3.3, AC3.4, AC3.5"
```
<!-- END_TASK_6 -->

<!-- END_SUBCOMPONENT_B -->
