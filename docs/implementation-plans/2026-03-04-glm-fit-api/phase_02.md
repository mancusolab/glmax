# GLM Fit API Implementation Plan

**Goal:** Reorganize model and inference boundaries so `GLM` is model specification and runtime internals live in a cohesive inference seam.

**Architecture:** Keep algorithm/solver/covariance implementations in `src/glmax/infer/` and relocate ownership boundaries without changing user-visible wrapper behavior yet.

**Tech Stack:** Python 3.11, JAX, Equinox, pytest.

**Scope:** 5 phases from original design (this file covers Phase 2).

**Codebase verified:** 2026-03-04 (America/Los_Angeles)

---

## Acceptance Criteria Coverage

This phase implements and tests:

### glm-fit-api.AC2: Inference Components Are Modular Under One Boundary
- **glm-fit-api.AC2.1 Success:** Fit algorithm, linear solver, and covariance strategy are independently swappable through `gx.fit(..., fitter=..., solver=..., covariance=...)`.
- **glm-fit-api.AC2.2 Success:** The refactor places these strategy components under one cohesive inference seam (not fragmented across unrelated taxonomy modules).

---

<!-- START_SUBCOMPONENT_A (tasks 1-4) -->
<!-- START_TASK_1 -->
### Task 1: Split runtime fit-state and user-facing result-state contracts

**Verifies:** glm-fit-api.AC2.2

**Files:**
- Create: `src/glmax/infer/state.py`
- Create: `src/glmax/infer/result.py`
- Modify: `src/glmax/infer/optimize.py:1-140`
- Modify: `src/glmax/infer/__init__.py:1-160`

**Implementation:**
- Move runtime-only state types (for example `IRLSState`) into `src/glmax/infer/state.py`.
- Define user-facing result-state contracts in `src/glmax/infer/result.py`.
- Re-export both from `src/glmax/infer/__init__.py`.

**Testing:**
- Ensure tests continue importing via public package APIs and fields remain compatible.

**Verification:**
- Run: `pytest -p no:capture tests/test_glm.py`
- Expected: no numerical regressions.

**Commit:** `refactor: split runtime and result state contracts in infer seam`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Restructure `GLM` class toward model-spec responsibility only

**Verifies:** glm-fit-api.AC2.2

**Files:**
- Modify: `src/glmax/glm.py:54-240`
- Modify: `src/glmax/fit.py:1-360`

**Implementation:**
- Remove direct ownership of orchestration helpers from `GLM` where possible, but do not convert `GLM.fit` delegation behavior in this phase.
- Keep `GLM` centered on family/link/default strategy metadata and helper compatibility surfaces.

**Testing:**
- Add structural tests that package-level API and model metadata remain coherent.

**Verification:**
- Run: `pytest -p no:capture tests/test_fit_api.py tests/test_glm.py`
- Expected: imports and behavior remain stable.

**Commit:** `refactor: align GLM with model-spec boundary`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Harden fully swappable strategy surface

**Verifies:** glm-fit-api.AC2.1, glm-fit-api.AC2.2

**Files:**
- Modify: `src/glmax/infer/__init__.py:1-180`
- Modify: `src/glmax/__init__.py:5-60`
- Modify: `tests/test_fit_api.py`

**Implementation:**
- Ensure fit algorithm (`fitter`), solver, covariance, and test-hook extension points are exported from coherent inference boundaries.
- Avoid taxonomy-only helper fragmentation.

**Testing:**
- Add explicit swap tests for all three strategy classes:
  - custom fitter via `gx.fit(..., fitter=...)`,
  - alternate solver via `gx.fit(..., solver=...)`,
  - alternate covariance estimator via `gx.fit(..., covariance=...)`.

**Verification:**
- Run: `pytest -p no:capture tests/test_fit_api.py`
- Expected: fitter/solver/covariance swap tests all pass.

**Commit:** `refactor: expose fully swappable inference strategy interfaces`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Document intentional phase boundary decisions

**Verifies:** glm-fit-api.AC2.2

**Files:**
- Modify: `docs/implementation-plans/2026-03-04-glm-fit-api/phase_02.md`

**Implementation:**
- Add a short “Phase boundary note” clarifying that state split is intentionally completed in Phase 2 per design contract.

**Testing:**
- N/A (documentation traceability step)

**Verification:**
- Run: `rg -n "Phase boundary note" docs/implementation-plans/2026-03-04-glm-fit-api/phase_02.md`
- Expected: note exists.

**Commit:** `docs: clarify phase boundary traceability for state split`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_A -->

## Phase Boundary Note
Runtime/result-state split is intentionally implemented in this phase to align with the design plan Phase 2 component "Result object split between dynamic runtime state and user-facing fit result representation."
This boundary is treated as complete for subsequent phases; later phases should consume these contracts rather than re-introducing mixed state ownership.
