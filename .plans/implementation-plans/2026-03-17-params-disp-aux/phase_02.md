# Params Disp Aux Implementation Plan

**Goal:** Split EDM dispersion from family-specific auxiliary parameters inside the family and `GLM` computation seams.

**Architecture:** Keep all family semantics inside `src/glmax/family/dist.py` and the `GLM` delegation surface in `src/glmax/glm.py`. This phase should change method signatures and docstrings in place instead of creating wrapper modules, and it should establish the `disp` versus `aux` split for every currently supported family before fit/infer plumbing is updated in Phase 3.

**Tech Stack:** Python 3.11, JAX, Equinox, jaxtyping, pytest

**Scope:** 4 phases from original design (phase 2 only in this file)

**Codebase verified:** 2026-03-17 12:12:51 PDT

---

## Acceptance Criteria Coverage

This phase implements and tests:

### `params-disp-aux.AC2`: family semantics split EDM dispersion from auxiliary parameters
- **`params-disp-aux.AC2.1` Success:** Gaussian and Gamma use `disp` as EDM dispersion and ignore `aux`.
- **`params-disp-aux.AC2.2` Success:** Poisson and Binomial canonicalize `disp` to `1.0` and ignore `aux`.
- **`params-disp-aux.AC2.3` Success:** Negative Binomial canonicalizes `disp` to `1.0` and uses `aux` as `alpha` in likelihood, variance, sampling, and fitting updates.
- **`params-disp-aux.AC2.4` Failure:** invalid NB `aux` values such as non-positive or non-finite `alpha` are rejected deterministically.
- **`params-disp-aux.AC2.5` Success:** family and `GLM` docstrings describe which of `disp` and `aux` each family uses, fixes, or ignores.

---

## Phase-by-Phase Implementation

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Rewrite family and GLM regression tests around `disp` versus `aux`

**Verifies:** `params-disp-aux.AC2.1`, `params-disp-aux.AC2.2`, `params-disp-aux.AC2.3`, `params-disp-aux.AC2.4`

**Files:**
- Modify: `tests/family/test_families.py:117`
- Modify: `tests/glm/test_glm.py:156`

**Implementation:**
Update the family numerics and GLM delegation tests so they stop treating Negative Binomial `disp` as `alpha`. Keep Gaussian and Gamma expectations on `params.disp`; keep Poisson and Binomial fixed at canonical `disp = 1.0` with no auxiliary state; add Negative Binomial assertions that `aux` is the fitted `alpha` used by likelihood, variance, gradient, and sampling paths. Replace the Statsmodels parity setup in `tests/glm/test_glm.py` so the reference Negative Binomial model reads `alpha` from `glm_state.params.aux`, not `glm_state.params.disp`.

Add failure coverage for invalid Negative Binomial `aux` values, including non-finite and non-positive inputs.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC2.1`: Gaussian and Gamma continue to use `disp` as EDM dispersion.
- `params-disp-aux.AC2.2`: Poisson and Binomial force canonical `disp = 1.0` and ignore auxiliary state.
- `params-disp-aux.AC2.3`: Negative Binomial uses `aux` for `alpha` across numerics and reference-package parity.
- `params-disp-aux.AC2.4`: invalid NB auxiliary values raise deterministic errors.

**Verification:**
Run: `pytest -p no:capture tests/family/test_families.py tests/glm/test_glm.py`
Expected: Family numerics and GLM delegation tests pass with the new semantics split.

**Commit:** `test: lock family disp aux semantics`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Implement the split family hooks in `src/glmax/family/dist.py`

**Verifies:** `params-disp-aux.AC2.1`, `params-disp-aux.AC2.2`, `params-disp-aux.AC2.3`, `params-disp-aux.AC2.4`, `params-disp-aux.AC2.5`

**Files:**
- Modify: `src/glmax/family/dist.py:25`

**Implementation:**
Extend the existing family API in place so the numerics surface can distinguish GLM dispersion from family-specific auxiliary state without adding a new module. The abstract family contract should expose canonicalization and validation hooks for both values, and the concrete family methods used by `GLM` should accept the split `(disp, aux)` inputs.

For concrete behavior:
- Gaussian and Gamma should continue to use `disp` as their EDM dispersion and ignore `aux`.
- Poisson and Binomial should canonicalize `disp` to `1.0` and ignore any provided auxiliary value.
- Negative Binomial should canonicalize `disp` to `1.0`, store `alpha` in `aux`, and use `aux` in `negloglikelihood`, `variance`, `sample`, `update_dispersion`/aux-update logic, and post-fit estimation.

Keep the work inside `src/glmax/family/dist.py`; do not split passive helpers into new files. Update docstrings in the same patch so the family contract explains which value each family uses, ignores, or fixes.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC2.1`: Gaussian and Gamma ignore `aux`.
- `params-disp-aux.AC2.2`: Poisson and Binomial canonicalize `disp` and forbid `aux`.
- `params-disp-aux.AC2.3`: Negative Binomial routes `alpha` through `aux`.
- `params-disp-aux.AC2.4`: invalid NB `aux` values are rejected.
- `params-disp-aux.AC2.5`: family docstrings describe the semantics correctly.

**Verification:**
Run: `pytest -p no:capture tests/family/test_families.py tests/glm/test_glm.py`
Expected: The family layer passes its numerics and delegation suite under the split contract.

**Commit:** `feat: split family dispersion and auxiliary semantics`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-3) -->
<!-- START_TASK_3 -->
### Task 3: Route the updated family semantics through the `GLM` helper surface

**Verifies:** `params-disp-aux.AC2.3`, `params-disp-aux.AC2.5`

**Files:**
- Modify: `src/glmax/glm.py:48`
- Modify: `tests/glm/test_glm.py:196`

**Implementation:**
Update the user-facing and kernel-facing `GLM` methods so they expose the split `disp`/`aux` contract consistently. `log_prob`, `sample`, `working_weights`, and any new parameter-canonicalization helpers should delegate through `self.family` but present the new two-scalar contract at the `GLM` boundary. Keep `scale(X, y, mu)` only as an implementation helper if it is still needed; do not let it remain the public source of truth for fitted dispersion in later phases.

Add or update GLM tests so the model methods match the family behavior under the new signatures and semantics.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC2.3`: Negative Binomial `GLM` helpers forward `aux` correctly.
- `params-disp-aux.AC2.5`: `GLM` docstrings and tests describe the split semantics accurately.

**Verification:**
Run: `pytest -p no:capture tests/family/test_families.py tests/glm/test_glm.py`
Expected: The `GLM` delegation layer matches the updated family contract.

**Commit:** `feat: route disp aux split through glm`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_B -->
