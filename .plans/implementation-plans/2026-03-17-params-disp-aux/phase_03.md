# Params Disp Aux Implementation Plan

**Goal:** Propagate the new parameter semantics through fitting, prediction, and inference without changing the grammar-first user workflow.

**Architecture:** Keep `glmax.fit` and `glmax.predict` as the public verbs in `src/glmax/_fit/fit.py`, keep IRLS state and canonicalization in `src/glmax/_fit/irls.py`, and update inference to consume the fitted carrier instead of recomputing dispersion from `family.scale(...)`. This phase should keep exports stable while changing how the existing fit/infer seams move `disp` and `aux`.

**Tech Stack:** Python 3.11, JAX, Equinox, jaxtyping, pytest

**Scope:** 4 phases from original design (phase 3 only in this file)

**Codebase verified:** 2026-03-17 12:12:51 PDT

---

## Acceptance Criteria Coverage

This phase implements and tests:

### `params-disp-aux.AC3`: fitting, prediction, and inference consume the correct parameter
- **`params-disp-aux.AC3.1` Success:** `fit(...)` returns canonical `Params(beta, disp, aux)` for every currently supported family.
- **`params-disp-aux.AC3.2` Success:** downstream inference uses `fitted.params.disp` as the covariance-scaling quantity `phi` and does not treat NB `aux` as GLM dispersion.
- **`params-disp-aux.AC3.3` Success:** `predict(...)` and GLM mean computations remain correct under the updated parameter carrier.
- **`params-disp-aux.AC3.4` Success:** Wald, Score, Fisher-information, and Huber-style inference outputs remain finite and shape-aligned under the new contract.
- **`params-disp-aux.AC3.5` Success:** supported-family regression checks continue to pass after the `disp`/`aux` split.

---

## Phase-by-Phase Implementation

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Update fit, predict, and inference regression tests for the canonical carrier

**Verifies:** `params-disp-aux.AC3.1`, `params-disp-aux.AC3.3`, `params-disp-aux.AC3.4`, `params-disp-aux.AC3.5`

**Files:**
- Modify: `tests/fit/test_fit.py:20`
- Modify: `tests/fit/test_predict.py:42`
- Modify: `tests/infer/test_infer.py:48`
- Modify: `tests/infer/test_stderr.py:41`
- Modify: `tests/infer/test_hypothesis.py:57`
- Modify: `tests/glm/test_glm.py:156`

**Implementation:**
Rewrite the regression suites so `fit(...)` is expected to return canonical `Params(beta, disp, aux)` for every supported family. Replace any assertions that Negative Binomial produces a positive `params.disp`; after the split, canonical NB fits should have `params.disp == 1.0` and a positive `params.aux`. Update the inference tests so the covariance scaling source is the stored fitted `params.disp`, not `family.scale(...)`, and keep the shape/finite-value checks around `InferenceResult` intact.

This task should also update the Statsmodels parity test in `tests/glm/test_glm.py` so the reference Negative Binomial `alpha` reads from `glm_state.params.aux`.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC3.1`: every supported family returns a canonical three-field carrier.
- `params-disp-aux.AC3.3`: prediction still returns stable finite means from the canonical carrier.
- `params-disp-aux.AC3.4`: inference result shapes and finiteness remain intact.
- `params-disp-aux.AC3.5`: supported-family regression checks still pass after the refactor.

**Verification:**
Run: `pytest -p no:capture tests/fit/test_fit.py tests/fit/test_predict.py tests/infer/test_infer.py tests/infer/test_stderr.py tests/infer/test_hypothesis.py tests/glm/test_glm.py`
Expected: The fit, predict, infer, and reference regression suites pass with the new carrier semantics.

**Commit:** `test: align fit predict infer with params aux`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Carry `aux` through IRLS and the public fit/predict verbs

**Verifies:** `params-disp-aux.AC3.1`, `params-disp-aux.AC3.3`, `params-disp-aux.AC3.5`

**Files:**
- Modify: `src/glmax/_fit/fit.py:24`
- Modify: `src/glmax/_fit/irls.py:23`
- Modify: `src/glmax/_fit/types.py:266`

**Implementation:**
Extend the fit pipeline so `_canonicalize_init(...)`, IRLS state, and the final `FitResult` all carry the canonical auxiliary value. `IRLSFitter.__call__` should initialize family-specific auxiliary state when it is absent from `init`, pass both `disp` and `aux` through `model.working_weights(...)` and `model.log_prob(...)`, and construct `Params(beta=..., disp=..., aux=...)` only after the model/family canonicalization hooks have run.

Keep `glmax.fit(...)` and `glmax.predict(...)` as thin public verbs; the new logic belongs in existing helpers and the IRLS kernel, not in extra wrapper modules.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC3.1`: fit returns canonical `Params(beta, disp, aux)`.
- `params-disp-aux.AC3.3`: predict works unchanged for the user-facing workflow.
- `params-disp-aux.AC3.5`: supported families still converge under the updated IRLS plumbing.

**Verification:**
Run: `pytest -p no:capture tests/fit/test_fit.py tests/fit/test_predict.py tests/glm/test_glm.py`
Expected: Fit and predict regression suites pass with the new IRLS parameter flow.

**Commit:** `feat: plumb aux through fit and predict`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-3) -->
<!-- START_TASK_3 -->
### Task 3: Make inference use the fitted `params.disp` as the `phi` source of truth

**Verifies:** `params-disp-aux.AC3.2`, `params-disp-aux.AC3.4`

**Files:**
- Modify: `src/glmax/_infer/stderr.py:41`
- Modify: `src/glmax/_infer/hyptest.py:55`
- Modify: `tests/infer/test_stderr.py:41`
- Modify: `tests/infer/test_hypothesis.py:57`

**Implementation:**
Update `FisherInfoError`, `HuberError`, and `ScoreTest` so they read `phi` from `fitted.params.disp` instead of recomputing it with `model.scale(X, y, mu)`. Keep the finite/positive guards in place, but make them validate the canonical fitted dispersion rather than the family helper. Negative Binomial must continue to treat `params.aux` as family-specific `alpha`; it must not leak into any covariance-scaling path.

Update inline comments and test formulas to reflect the new source of truth for `phi`.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC3.2`: inference covariance scaling uses `fitted.params.disp`.
- `params-disp-aux.AC3.4`: Wald, Score, Fisher, and Huber outputs remain finite and correctly shaped.

**Verification:**
Run: `pytest -p no:capture tests/infer/test_infer.py tests/infer/test_stderr.py tests/infer/test_hypothesis.py`
Expected: The inference suite passes with `params.disp` as the only fitted `phi` source.

**Commit:** `feat: use fitted dispersion in inference`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->
