# Params Disp Aux Implementation Plan

**Goal:** Introduce `Params(beta, disp, aux)` as the canonical carrier without changing the top-level grammar workflow.

**Architecture:** Keep the carrier, warm-start validation, and family-boundary logic in the existing `_fit` and `glm` seams. This phase should not create new modules; it should extend the current `Params`/`FitResult`/`FittedGLM` contracts in place and lock the behavior with package, fit, and GLM tests before family-specific semantics are split in later phases.

**Tech Stack:** Python 3.11, JAX, Equinox, jaxtyping, pytest

**Scope:** 4 phases from original design (phase 1 only in this file)

**Codebase verified:** 2026-03-17 12:12:51 PDT

---

## Acceptance Criteria Coverage

This phase implements and tests:

### `params-disp-aux.AC1`: `Params` carries one stable meaning per field
- **`params-disp-aux.AC1.1` Success:** `Params` stores `beta`, `disp`, and `aux`, and remains a valid pytree and warm-start carrier.
- **`params-disp-aux.AC1.2` Success:** `beta` remains the coefficient vector, `disp` remains the GLM/EDM dispersion scalar, and `aux` remains the optional family-specific scalar.
- **`params-disp-aux.AC1.3` Failure:** non-inexact or non-scalar `disp`/`aux` values raise deterministic validation errors.
- **`params-disp-aux.AC1.4` Success:** families that do not use an auxiliary parameter ignore any provided `aux` and canonicalize it to `None`.
- **`params-disp-aux.AC1.5` Success:** warm-start paths accept canonical `Params(beta, disp, aux)` and preserve values through `fit(...)` and `infer(...)`.

---

## Phase-by-Phase Implementation

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Update package and fit contract tests for the three-field `Params` carrier

**Verifies:** `params-disp-aux.AC1.1`, `params-disp-aux.AC1.2`, `params-disp-aux.AC1.3`, `params-disp-aux.AC1.5`

**Files:**
- Modify: `tests/package/test_api.py:25`
- Modify: `tests/package/test_grammar.py:37`
- Modify: `tests/data/test_glmdata.py:77`
- Modify: `tests/fit/test_fit.py:20`
- Modify: `tests/fit/test_predict.py:14`

**Implementation:**
Update the existing contract suites so every direct `Params(...)` construction uses `beta`, `disp`, and `aux`. Rewrite the pytree assertions to expect three leaves instead of two, keep `Params` as a `NamedTuple`, and add deterministic public-boundary failures for non-numeric, non-inexact, and non-scalar `aux` values alongside the existing `beta` and `disp` checks. Update the data-layer schema check that currently asserts `["beta", "disp"]` so it locks the new carrier shape instead of failing late during execution. Extend the warm-start roundtrip assertions so `fit(...)` followed by `fit(..., init=fitted.params)` preserves canonical `disp` and `aux`.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC1.1`: `Params` remains a pytree and warm-start carrier after adding `aux`.
- `params-disp-aux.AC1.2`: contract tests still pin the semantic roles of `beta`, `disp`, and `aux`.
- `params-disp-aux.AC1.3`: invalid `aux` shapes and dtypes fail with deterministic errors at public boundaries.
- `params-disp-aux.AC1.5`: warm-start roundtrips keep canonical values intact.

**Verification:**
Run: `pytest -p no:capture tests/package/test_api.py tests/package/test_grammar.py tests/data/test_glmdata.py tests/fit/test_fit.py tests/fit/test_predict.py`
Expected: All package and fit boundary tests pass with the new three-field `Params` contract.

**Commit:** `test: pin params aux carrier contract`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Extend `_fit` carrier validation and public noun plumbing for `aux`

**Verifies:** `params-disp-aux.AC1.1`, `params-disp-aux.AC1.2`, `params-disp-aux.AC1.3`, `params-disp-aux.AC1.5`

**Files:**
- Modify: `src/glmax/_fit/types.py:15`
- Modify: `src/glmax/_fit/fit.py:24`

**Implementation:**
Add `aux` to `Params` and keep the carrier immutable. Update `FitResult.__check_init__` so both `disp` and `aux` are validated as scalar inexact values when present, and extend `_canonicalize_init(...)` to return `(beta, disp, aux)` instead of dropping family-specific state. Keep `glmax.fit(...)` and `glmax.predict(...)` validating `Params` at the public boundary, and make `predict(...)` canonicalize the full carrier even though prediction still only uses `beta` for `eta -> mu`.

Do not create a new carrier module. Keep the carrier in `src/glmax/_fit/types.py`, and keep `glmax.fit`/`glmax.predict` as the only public verb entrypoints in `src/glmax/_fit/fit.py`.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC1.1`: the concrete `Params`, `FitResult`, and `FittedGLM` carriers still work as pytrees/nouns.
- `params-disp-aux.AC1.2`: the new field is wired without changing the meanings of `beta` or `disp`.
- `params-disp-aux.AC1.3`: invalid `aux` values fail in `_canonicalize_init(...)` and `FitResult` validation.
- `params-disp-aux.AC1.5`: the public verbs accept and preserve canonical `Params(beta, disp, aux)`.

**Verification:**
Run: `pytest -p no:capture tests/package/test_api.py tests/package/test_grammar.py tests/data/test_glmdata.py tests/fit/test_fit.py tests/fit/test_predict.py`
Expected: The updated carrier and public verbs pass the narrowed contract suite.

**Commit:** `feat: add aux to params carrier`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-3) -->
<!-- START_TASK_3 -->
### Task 3: Add family-aware parameter canonicalization hooks to `GLM`

**Verifies:** `params-disp-aux.AC1.4`, `params-disp-aux.AC1.5`

**Files:**
- Modify: `src/glmax/glm.py:48`
- Modify: `tests/glm/test_glm.py:196`

**Implementation:**
Add explicit `GLM` methods that let later fit and inference kernels canonicalize and validate `(disp, aux)` through the model boundary instead of reaching into `model.family` directly. Keep these hooks in `src/glmax/glm.py`; do not introduce a new semantics file. At minimum, Phase 1 needs a model-level path that canonicalizes unused `aux` inputs to `None` for families without auxiliary state and returns canonical warm-start values for supported families.

Add GLM-level regression tests that pin this delegation boundary so Phase 2 can layer the family-specific semantics on top of it without moving the API surface again.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC1.4`: families without auxiliary state ignore provided `aux` and canonicalize it to `None`.
- `params-disp-aux.AC1.5`: warm-start canonicalization is exposed through the `GLM` boundary rather than ad hoc kernel logic.

**Verification:**
Run: `pytest -p no:capture tests/package/test_api.py tests/package/test_grammar.py tests/data/test_glmdata.py tests/fit/test_fit.py tests/fit/test_predict.py tests/glm/test_glm.py`
Expected: Package, fit, predict, and GLM contract tests all pass with the new parameter canonicalization seam.

**Commit:** `feat: add glm param canonicalization hooks`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->
