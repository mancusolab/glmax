# Test Requirements — Inferrer Abstraction

**Plan:** `docs/implementation-plans/2026-03-16-inferrer-abstraction/`
**Design:** `docs/design-plans/2026-03-16-inferrer-abstraction.md`
**Date:** 2026-03-16

---

## Summary

This document maps every acceptance criterion from the inferrer-abstraction design to either an automated test or a documented human-verification step. All 16 sub-criteria across 5 ACs have automated coverage. No criteria are deferred to human-only verification; the human test plan is provided as supplementary end-to-end validation.

---

## Acceptance Criteria to Test Mapping

### AC1 — `InferenceResult.stat` field rename

**Phase:** 1
**Scope:** Breaking rename of `InferenceResult.z` to `InferenceResult.stat` in isolation.

#### AC1.1 — `InferenceResult` instances expose `.stat`, not `.z`

- **Test type:** Unit
- **Expected test file:** `tests/test_infer_verbs.py`
- **Description:** The existing assertion `assert inferred.z.shape == fitted.params.beta.shape` (line 56) is updated to `assert inferred.stat.shape == fitted.params.beta.shape`. After Phase 1 task 2, any access of `.z` on an `InferenceResult` must raise `AttributeError` because `NamedTuple` field order and names are fixed at class definition. The updated assertion positively confirms `.stat` is present.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_infer_returns_inference_result_without_refitting`

#### AC1.2 — Accessing `.z` on an `InferenceResult` raises `AttributeError`

- **Test type:** Unit
- **Expected test file:** `tests/test_infer_verbs.py`
- **Description:** A separate assertion (or a dedicated test) calls `getattr(inferred, "z", sentinel)` and asserts the sentinel is returned, OR calls `inferred.z` and asserts `AttributeError` is raised. This is the failure-mode verification for the rename. The implementation plan relies on the `NamedTuple` semantics: once `.z` is removed from the definition, attribute lookup raises `AttributeError` naturally. A `pytest.raises(AttributeError)` block makes this explicit and prevents regression.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_infer_returns_inference_result_without_refitting`
- **Rationalization note:** Phase 1 task 2 instructs a `grep -rn "\.z\b" tests/` sweep to confirm no stale `.z` accesses remain. That grep is a pre-commit check, not an automated test. The explicit `AttributeError` assertion in the test file is what provides ongoing regression protection. If no such assertion is added, AC1.2 is covered only by absence of stale references, which is weaker. The test file should include `with pytest.raises(AttributeError): _ = inferred.z` or equivalent.

#### AC1.3 — `.stat` shape matches `.params.beta` shape `(p,)`

- **Test type:** Unit
- **Expected test file:** `tests/test_infer_verbs.py`
- **Description:** `assert inferred.stat.shape == fitted.params.beta.shape` (the renamed assertion from AC1.1 coverage). Shape equality is confirmed by the same assertion; no additional test is needed.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_infer_returns_inference_result_without_refitting`

---

### AC2 — `WaldInferrer` correctness

**Phase:** 2
**Scope:** New `tests/test_inferrers.py`; `WaldInferrer` in `src/glmax/infer/inferrer.py`.

#### AC2.1 — `WaldInferrer()(fitted, stderr)` returns same p-values as pre-refactor `infer()` to float64 precision

- **Test type:** Unit / numeric regression
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_wald_inferrer_matches_legacy_infer`
- **Description:** Calls `legacy_infer(fitted)` (imported as `from glmax.infer.inference import infer as legacy_infer` before Phase 3 wires the new signature) and `WaldInferrer()(fitted, FisherInfoError())`. Asserts `jnp.allclose(wald_result.stat, legacy_result.z, atol=1e-12)`, `jnp.allclose(wald_result.se, legacy_result.se, atol=1e-12)`, `jnp.allclose(wald_result.p, legacy_result.p, atol=1e-12)`. This is the primary numeric regression guard.
- **Rationalization note:** `legacy_infer` refers to the pre-Phase-3 `infer()` — the one that still hard-codes Wald logic. After Phase 3 wires the delegation, `glmax.infer(fitted)` is the equivalent reference. The test may need the legacy reference updated to `glmax.infer(fitted, inferrer=WaldInferrer())` post-Phase-3, or can simply compare two explicit `WaldInferrer()` calls. Either is valid; the numeric constraint (`atol=1e-12`) must be preserved.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_wald_inferrer_matches_legacy_infer`

#### AC2.2 — Gaussian family uses t-distribution; all other families use standard normal

- **Test type:** Unit
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_wald_inferrer_gaussian_uses_t_distribution`
- **Description:** Indirect verification via regression against `legacy_infer()`: since the legacy `wald_test()` function already implements the Gaussian/t vs. normal branch, and AC2.1 confirms numeric identity, AC2.2 is automatically co-verified. An explicit cross-family test (Gaussian vs. Poisson) comparing p-values for the same statistic strengthens this: the Gaussian t-based p-value and the normal-based p-value differ measurably for small `n` (df < 30), so the two families must not produce identical results.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_wald_inferrer_gaussian_uses_t_distribution`

#### AC2.3 — `WaldInferrer` calls `stderr(fitted)` internally and uses the resulting covariance to compute SE

- **Test type:** Unit / integration
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_wald_inferrer_uses_injected_stderr`
- **Description:** Defines `ConstantCovStdErr(AbstractStdErrEstimator)` (no `strict=True`, no instance attributes) returning `jnp.eye(2) * 4.0`. Calls `WaldInferrer()(fitted_2col, ConstantCovStdErr())`. Asserts `jnp.allclose(result.se, jnp.array([2.0, 2.0]))`. Confirms `stderr` was called and its covariance diagonal was sqrt-extracted.
- **Rationalization note:** The stub must use a two-column `X` fixture (matching `_make_fitted` in `test_inferrers.py` which uses shape `(4, 2)`) so the covariance matrix is `(2, 2)`. The single-column `_make_fitted` in `test_infer_verbs.py` produces a `(1, 1)` covariance — this distinction is important for the `jnp.eye(2) * 4.0` stub to be consistent.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_wald_inferrer_uses_injected_stderr`

#### AC2.4 — Non-`FittedGLM` first arg raises `TypeError`

- **Test type:** Unit / contract
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_wald_inferrer_rejects_non_fitted_glm`
- **Description:** Calls `WaldInferrer()(object(), FisherInfoError())`. Asserts `pytest.raises(TypeError)`.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_wald_inferrer_rejects_non_fitted_glm`

---

### AC3 — `ScoreInferrer` correctness

**Phase:** 2
**Scope:** `tests/test_inferrers.py`; `ScoreInferrer` in `src/glmax/infer/inferrer.py`.

#### AC3.1 — `ScoreInferrer()(fitted, stderr)` returns `InferenceResult` with `stat` finite, `p` in `[0,1]`, `se` all-NaN

- **Test type:** Unit / numeric contract
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_score_inferrer_returns_valid_result`
- **Description:** Calls `ScoreInferrer()(fitted_gaussian, FisherInfoError())`. Asserts: `isinstance(result, InferenceResult)`, `bool(jnp.all(jnp.isfinite(result.stat)))`, `bool(jnp.all((result.p >= 0.0) & (result.p <= 1.0)))`, `bool(jnp.all(jnp.isnan(result.se)))`.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_score_inferrer_returns_valid_result`

#### AC3.2 — `ScoreInferrer` does not call `stderr`

- **Test type:** Unit / behavioral
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_score_inferrer_does_not_call_stderr`
- **Description:** Defines `RaisingStdErr(AbstractStdErrEstimator)` (no `strict=True`) whose `__call__` raises `RuntimeError("stderr should not be called")`. Calls `ScoreInferrer()(fitted, RaisingStdErr())`. Asserts the call completes without exception.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_score_inferrer_does_not_call_stderr`

#### AC3.3 — `stat` shape matches `(p,)` for all supported families

- **Test type:** Unit / parametrize
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_score_inferrer_stat_shape_matches_beta`
- **Description:** Parametrized over `[Gaussian(), Poisson(), Binomial()]`. For each, builds the appropriate fitted model (Binomial requires integer `y`) and asserts `result.stat.shape == fitted.params.beta.shape`. `NegativeBinomial` is explicitly excluded: `GLM.calc_eta_and_dispersion` calls `self.fit` internally for NB (known fragile self-recursion, documented in MEMORY.md known issues), and exposing that risk in a new test suite is out of scope for this plan.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_score_inferrer_stat_shape_matches_beta`

#### AC3.4 — Gaussian family matches the direct MLE-point score-style formula

- **Test type:** Unit / numeric
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_score_inferrer_gaussian_p_values_are_valid`
- **Description:** Calls `ScoreInferrer()` on a Gaussian fitted model. Asserts p-values are in `(0, 1]` and finite. Additionally computes the direct MLE-point score-style statistic from `score_residual`, `glm_wt`, `phi`, and the Fisher-information diagonal, then asserts `jnp.allclose` on both `stat` and the corresponding normal-reference p-values. This upgrades the "is it finite?" check from AC3.1 to a correctness check for the Gaussian case without asserting score-vs-Wald equivalence.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_score_inferrer_gaussian_p_values_are_valid`

#### AC3.5 — Degenerate scale or information fails deterministically

- **Test type:** Unit / failure-path
- **Expected test file:** `tests/test_inferrers.py`
- **Test name:** `test_score_inferrer_rejects_degenerate_gaussian_scale`
- **Description:** Builds a perfect-fit Gaussian model where `family.scale(X, y, mu)` collapses to zero. Asserts `ScoreInferrer()` raises `ValueError` instead of returning NaN statistics or p-values. This protects the contract against silent invalid outputs on degenerate fits.
- **Command:** `pytest -p no:capture tests/test_inferrers.py::test_score_inferrer_rejects_degenerate_gaussian_scale`

---

### AC4 — `infer()` signature and delegation

**Phase:** 3
**Scope:** `tests/test_infer_verbs.py` (new tests appended), `tests/test_fit_api.py` (updated assertions).

#### AC4.1 — `infer(fitted)` with no extra args produces same result as pre-refactor

- **Test type:** Integration / regression
- **Expected test file:** `tests/test_infer_verbs.py`
- **Test name:** `test_infer_default_inferrer_matches_explicit_wald_inferrer`
- **Description:** Calls `glmax.infer(fitted)` and `glmax.infer(fitted, inferrer=WaldInferrer())`. Asserts `jnp.allclose` on `stat`, `se`, and `p`. This confirms `DEFAULT_INFERRER = WaldInferrer()` and that passing no `inferrer` argument routes to it.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_infer_default_inferrer_matches_explicit_wald_inferrer`

#### AC4.2 — `infer(fitted, inferrer=ScoreInferrer())` routes to `ScoreInferrer`

- **Test type:** Integration
- **Expected test file:** `tests/test_infer_verbs.py`
- **Test name:** `test_infer_routes_to_score_inferrer_when_specified`
- **Description:** Calls `glmax.infer(fitted, inferrer=ScoreInferrer())`. Asserts `isinstance(result, InferenceResult)`, `jnp.all(jnp.isnan(result.se))`, `jnp.all(jnp.isfinite(result.stat))`, p-values in `[0, 1]`, `result.stat.shape == fitted.params.beta.shape`. The `se` all-NaN distinguishes a `ScoreInferrer` result from a `WaldInferrer` result.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_infer_routes_to_score_inferrer_when_specified`

#### AC4.3 — `infer(fitted, stderr=HuberError())` passes `HuberError` into `WaldInferrer`

- **Test type:** Integration / behavioral
- **Expected test file:** `tests/test_infer_verbs.py`
- **Test name:** `test_infer_passes_stderr_into_wald_inferrer`
- **Description:** Uses a `CountingStdErr(AbstractStdErrEstimator)` (no `strict=True`, no instance attributes; call count tracked via closed-over dict) returning a known `1×1` covariance. Calls `glmax.infer(fitted, stderr=CountingStdErr())`. Asserts `call_count["n"] == 1` and `jnp.allclose(result.se, jnp.array([2.0]))`. This verifies that the `stderr` keyword argument is threaded through the shell into `WaldInferrer`.
- **Rationalization note:** The `_make_fitted()` fixture in `test_infer_verbs.py` uses a single-column `X`, giving `beta` shape `(1,)`. The `CountingStdErr` therefore returns `jnp.array([[4.0]])` (shape `1×1`), yielding `se = [2.0]`.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_infer_passes_stderr_into_wald_inferrer`

#### AC4.4 — `infer(fitted, inferrer=object())` raises `TypeError`

- **Test type:** Unit / contract
- **Expected test file:** `tests/test_infer_verbs.py`
- **Test name:** Addition to `test_infer_rejects_invalid_model_and_result_contracts`
- **Description:** Adds `with pytest.raises(TypeError, match="AbstractInferrer"): glmax.infer(fitted, inferrer=object())` to the existing contract test. Confirms the `isinstance(inferrer, AbstractInferrer)` guard in the updated `infer()` body fires before delegation.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_infer_rejects_invalid_model_and_result_contracts`

#### AC4.5 — `infer(fitted, stderr=object())` raises `TypeError`

- **Test type:** Unit / contract
- **Expected test file:** `tests/test_infer_verbs.py`
- **Test name:** Existing `test_infer_rejects_invalid_model_and_result_contracts` (lines 82-83)
- **Description:** The existing assertion `with pytest.raises(TypeError, match="AbstractStdErrEstimator"): glmax.infer(fitted, stderr=object())` already covers this criterion. After the Phase 3 signature change, `stderr` moves from second to third parameter but remains keyword-accessible; the existing test passes `stderr=object()` as a keyword argument, so it continues to exercise the correct guard path without modification.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_infer_rejects_invalid_model_and_result_contracts`

---

### AC5 — Public surface exports

**Phase:** 3
**Scope:** `tests/test_infer_verbs.py` (new import tests), `tests/test_fit_api.py` (updated `__all__` assertion and signature test).

#### AC5.1 — `from glmax import AbstractInferrer, WaldInferrer, ScoreInferrer` succeeds

- **Test type:** Integration / import
- **Expected test file:** `tests/test_infer_verbs.py`
- **Test name:** `test_inferrer_types_are_importable_from_top_level_glmax`
- **Description:** Executes `from glmax import AbstractInferrer, WaldInferrer, ScoreInferrer` inside the test body (avoiding module-level import to isolate the assertion from other import errors). Asserts each name is not `None`. This confirms `src/glmax/__init__.py` exports all three names at runtime.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_inferrer_types_are_importable_from_top_level_glmax`

#### AC5.2 — `from glmax import AbstractStdErrEstimator, FisherInfoError, HuberError` succeeds

- **Test type:** Integration / import
- **Expected test file:** `tests/test_infer_verbs.py`
- **Test name:** `test_stderr_types_are_importable_from_top_level_glmax`
- **Description:** Same pattern as AC5.1 for the three SE estimator names. These were previously not exported from the top-level `glmax` namespace; this test confirms the Phase 3 addition to `__init__.py` is correct.
- **Command:** `pytest -p no:capture tests/test_infer_verbs.py::test_stderr_types_are_importable_from_top_level_glmax`

#### AC5.3 — All six names appear in `glmax.__all__`

- **Test type:** Unit / surface contract
- **Expected test file:** `tests/test_fit_api.py`
- **Test name:** `test_top_level_exports_are_canonical_nouns_and_verbs`
- **Description:** The existing `assert set(glmax.__all__) == {...}` exhaustive set-equality test is updated to include all six new names: `AbstractInferrer`, `WaldInferrer`, `ScoreInferrer`, `AbstractStdErrEstimator`, `FisherInfoError`, `HuberError`. Set equality (not subset) means any accidental addition or removal also fails the test. Expected total: 19 items in `__all__`.
- **Command:** `pytest -p no:capture tests/test_fit_api.py::test_top_level_exports_are_canonical_nouns_and_verbs`

---

## Supporting Infrastructure Tests

These tests exist in the current suite and remain in scope across phases; they are not new ACs but are affected by the refactor and must stay green.

| Test | File | Why it matters |
|------|------|----------------|
| `test_infer_returns_inference_result_without_refitting` | `test_infer_verbs.py` | Baseline shape contract; `.z` → `.stat` rename touches this directly |
| `test_infer_uses_injected_stderr_estimator` | `test_infer_verbs.py` | `stderr` injection works end-to-end through the shell; must survive Phase 3 shell rewrite |
| `test_infer_rejects_invalid_fit_artifacts_deterministically` | `test_infer_verbs.py` | `validate_fit_result` still fires via the inferrer; must remain green for both `WaldInferrer` and `ScoreInferrer` paths |
| `test_infer_never_calls_fit_or_irls` | `test_infer_verbs.py` | `infer()` must not trigger IRLS regardless of which inferrer is used |
| `test_infer_signature_matches_canonical_surface` | `test_fit_api.py` | Parameter list changes from `["fitted", "stderr"]` to `["fitted", "inferrer", "stderr"]`; default for `inferrer` must be `None` |
| `test_infer_shims_are_not_publicly_reexported` | `test_fit_api.py` | `glmax.infer` namespace must NOT gain `AbstractInferrer`, `WaldInferrer`, etc. as runtime attributes; only `glmax.*` exports them |

---

## Human Verification

No acceptance criteria require human-only verification. All 16 sub-criteria are covered by automated tests. The following human checks are supplementary and recommended prior to merging the final phase commit.

### Why No Criteria Are Human-Only

- The rename (AC1) is structurally verified by `NamedTuple` attribute lookup semantics.
- Numeric correctness (AC2.1, AC3.4) is verified by float64 regression assertions.
- Behavioral contracts (AC2.4, AC3.2, AC4.4, AC4.5) are verified by `pytest.raises` and stub patterns.
- Public surface (AC5) is verified by exhaustive `__all__` set-equality and import assertions.

### Supplementary Human Checks (Post-Phase-3)

These steps verify overall coherence and catch issues that individual unit tests cannot, such as import-time side effects, REPL experience, and documentation accuracy.

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Run `python -c "import glmax; print(glmax.__all__)"` in a clean virtual environment | Prints a list of 19 names including all six new exports; no import errors |
| 2 | In a Python REPL: `import glmax; r = glmax.infer(glmax.fit(glmax.specify(family=glmax.GLMData.__class__), ...))` — use any valid Gaussian dataset | `InferenceResult` with `.stat` (not `.z`); no `AttributeError`; p-values in `(0, 1)` |
| 3 | Access `.z` on the result from step 2 | `AttributeError: InferenceResult object has no attribute 'z'` |
| 4 | Call `glmax.infer(fitted, inferrer=glmax.ScoreInferrer())` and inspect `result.se` | Array of NaN values; `result.stat` and `result.p` are finite and in range |
| 5 | Confirm `glmax.infer` introspection: `import inspect; inspect.signature(glmax.infer)` | `(fitted, inferrer=None, stderr=<FisherInfoError ...>)` — `inferrer` default is `None`, not a constructed object |
| 6 | Verify `glmax.infer` module origin: `glmax.infer.__module__` | `'glmax.infer'` (delegates through shell; not `glmax.infer.inference`) |
| 7 | Run `pytest -p no:capture tests` from the worktree root | All tests pass; count is 217 (baseline) plus new tests from `test_inferrers.py` and the appended tests in `test_infer_verbs.py`; zero failures, zero errors |

---

## Traceability Matrix

| AC ID | Phase | Task IDs | Test File | Test Name | Command Fragment |
|-------|-------|----------|-----------|-----------|-----------------|
| AC1.1 | 1 | P1-T2 | `test_infer_verbs.py` | `test_infer_returns_inference_result_without_refitting` | `tests/test_infer_verbs.py::test_infer_returns_inference_result_without_refitting` |
| AC1.2 | 1 | P1-T2 | `test_infer_verbs.py` | `test_infer_returns_inference_result_without_refitting` | same — requires explicit `AttributeError` assertion on `.z` |
| AC1.3 | 1 | P1-T2 | `test_infer_verbs.py` | `test_infer_returns_inference_result_without_refitting` | same |
| AC2.1 | 2 | P2-T1, P2-T2, P2-T3 | `test_inferrers.py` | `test_wald_inferrer_matches_legacy_infer` | `tests/test_inferrers.py::test_wald_inferrer_matches_legacy_infer` |
| AC2.2 | 2 | P2-T1, P2-T2 | `test_inferrers.py` | `test_wald_inferrer_gaussian_uses_t_distribution` | `tests/test_inferrers.py::test_wald_inferrer_gaussian_uses_t_distribution` |
| AC2.3 | 2 | P2-T1, P2-T2 | `test_inferrers.py` | `test_wald_inferrer_uses_injected_stderr` | `tests/test_inferrers.py::test_wald_inferrer_uses_injected_stderr` |
| AC2.4 | 2 | P2-T1, P2-T2 | `test_inferrers.py` | `test_wald_inferrer_rejects_non_fitted_glm` | `tests/test_inferrers.py::test_wald_inferrer_rejects_non_fitted_glm` |
| AC3.1 | 2 | P2-T4, P2-T5, P2-T6 | `test_inferrers.py` | `test_score_inferrer_returns_valid_result` | `tests/test_inferrers.py::test_score_inferrer_returns_valid_result` |
| AC3.2 | 2 | P2-T4, P2-T5 | `test_inferrers.py` | `test_score_inferrer_does_not_call_stderr` | `tests/test_inferrers.py::test_score_inferrer_does_not_call_stderr` |
| AC3.3 | 2 | P2-T4, P2-T5 | `test_inferrers.py` | `test_score_inferrer_stat_shape_matches_beta` | `tests/test_inferrers.py::test_score_inferrer_stat_shape_matches_beta` |
| AC3.4 | 2 | P2-T4, P2-T5 | `test_inferrers.py` | `test_score_inferrer_gaussian_p_values_are_valid` | `tests/test_inferrers.py::test_score_inferrer_gaussian_p_values_are_valid` |
| AC3.5 | 2 | P2-T4, P2-T5 | `test_inferrers.py` | `test_score_inferrer_rejects_degenerate_gaussian_scale` | `tests/test_inferrers.py::test_score_inferrer_rejects_degenerate_gaussian_scale` |
| AC4.1 | 3 | P3-T1, P3-T2, P3-T5 | `test_infer_verbs.py` | `test_infer_default_inferrer_matches_explicit_wald_inferrer` | `tests/test_infer_verbs.py::test_infer_default_inferrer_matches_explicit_wald_inferrer` |
| AC4.2 | 3 | P3-T1, P3-T2, P3-T5 | `test_infer_verbs.py` | `test_infer_routes_to_score_inferrer_when_specified` | `tests/test_infer_verbs.py::test_infer_routes_to_score_inferrer_when_specified` |
| AC4.3 | 3 | P3-T1, P3-T2, P3-T5 | `test_infer_verbs.py` | `test_infer_passes_stderr_into_wald_inferrer` | `tests/test_infer_verbs.py::test_infer_passes_stderr_into_wald_inferrer` |
| AC4.4 | 3 | P3-T1, P3-T5 | `test_infer_verbs.py` | `test_infer_rejects_invalid_model_and_result_contracts` | `tests/test_infer_verbs.py::test_infer_rejects_invalid_model_and_result_contracts` |
| AC4.5 | 3 | P3-T1 (existing) | `test_infer_verbs.py` | `test_infer_rejects_invalid_model_and_result_contracts` | same — existing assertion, no new test needed |
| AC5.1 | 3 | P3-T3, P3-T5 | `test_infer_verbs.py` | `test_inferrer_types_are_importable_from_top_level_glmax` | `tests/test_infer_verbs.py::test_inferrer_types_are_importable_from_top_level_glmax` |
| AC5.2 | 3 | P3-T3, P3-T5 | `test_infer_verbs.py` | `test_stderr_types_are_importable_from_top_level_glmax` | `tests/test_infer_verbs.py::test_stderr_types_are_importable_from_top_level_glmax` |
| AC5.3 | 3 | P3-T3, P3-T4 | `test_fit_api.py` | `test_top_level_exports_are_canonical_nouns_and_verbs` | `tests/test_fit_api.py::test_top_level_exports_are_canonical_nouns_and_verbs` |

---

## Rationalization Notes Against Implementation Decisions

### `NegativeBinomial` exclusion from AC3.3

The `NegativeBinomial` family is excluded from the `ScoreInferrer` shape parametrize because `GLM.calc_eta_and_dispersion` contains a known fragile self-recursion (`self.fit` is called internally during `scale()` estimation for NB). This is documented in MEMORY.md known issues. Exercising `ScoreInferrer` on NB in an automated test would either silently hide this risk or introduce a flaky test. The design's risk register (R3 scope note) accepts this limitation. The exclusion is documented in the test comment.

### `AbstractStdErrEstimator` with `strict=True` and stub subclasses

The design notes (phase_02.md, phase_03.md) explicitly flag that `AbstractStdErrEstimator` is `eqx.Module, strict=True`. All stub subclasses used in tests must omit `strict=True` and must not declare new instance attributes. Closed-over mutable dicts (e.g., `call_count = {"n": 0}`) are the correct pattern for recording state in these stubs. This is already established in `test_infer_verbs.py`'s `RecordingStdErr` stub and is carried forward to all new stubs.

### `glmax.infer` namespace encapsulation (AC5.3 / `test_infer_shims_are_not_publicly_reexported`)

The design decision (phase_03.md task 2 note) is that `AbstractInferrer`, `WaldInferrer`, and `ScoreInferrer` must NOT be runtime attributes of `glmax.infer`. They are added to `TYPE_CHECKING` only in `infer/__init__.py`. The existing `test_infer_shims_are_not_publicly_reexported` test in `test_fit_api.py` guards this; it does not need to be updated for Phase 3. Any attempt to add inferrer types as runtime `glmax.infer` attributes to "make them more discoverable" would break this test.

### `inferrer=None` default and lazy resolution

The `infer()` signature uses `inferrer=None` (not `inferrer=DEFAULT_INFERRER`) to avoid importing `inferrer.py` at module load time, preventing a circular import (`inference.py` → `inferrer.py` → `inference.py`). The Phase 3 signature test in `test_fit_api.py` must explicitly assert `sig.parameters["inferrer"].default is None` — not just that the parameter exists. This is a correctness constraint on the lazy-resolution pattern, not a style preference.

### `WaldInferrer` numeric regression tolerance

The `atol=1e-12` tolerance in `test_wald_inferrer_matches_legacy_infer` is intentional. The `WaldInferrer` implementation in phase_02.md is a direct copy of the pre-refactor `infer()` Wald computation (same JAX operations, same operand order). Float64 identity is expected; any deviation above `1e-12` would indicate an unintended algorithmic difference, not merely floating-point accumulation noise.

### `ScoreInferrer` MLE-point formula and degenerate-fit guard (AC3.4, AC3.5)

`ScoreInferrer` is validated against its direct MLE-point formula rather than against Wald equivalence. This matches the accepted design for this branch: `ScoreInferrer` is an MLE-point score-style diagnostic, not a restricted-model Rao score test. Degenerate fits with zero scale or zero information must fail deterministically with `ValueError` instead of emitting NaN summaries.
