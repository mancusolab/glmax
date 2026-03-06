# Infer Seam Consolidation Implementation Plan

**Goal:** Freeze the intended public grammar surface and characterize the duplicated `infer/` internals before consolidation.

**Architecture:** Keep the top-level grammar nouns in `src/glmax/contracts.py` and strategy objects in `src/glmax/infer/`. Treat duplicated `infer/*` modules as refactor targets, not public contracts to preserve.

**Tech Stack:** Python 3.11, JAX, Equinox, Lineax, pytest.

**Scope:** 3 phases total (this file covers Phase 1).

**Codebase verified:** 2026-03-06 (America/Los_Angeles)

---

## Phase Intent

This phase does not remove code yet. It establishes the behavioral guardrails needed to safely collapse the duplicated infer seam.

Verified current-state inputs to this phase:

- Canonical public nouns live in `src/glmax/contracts.py`.
- Canonical fit flow is `src/glmax/fit.py -> src/glmax/glm.py -> src/glmax/infer/optimize.py`.
- `src/glmax/infer/contracts.py` and `src/glmax/infer/solvers.py` duplicate solver boundaries already present in `src/glmax/infer/solve.py`.
- `src/glmax/infer/fitter.py` references `_run_default_pipeline`, which no longer exists.
- `src/glmax/infer/result.py` only serves the stale `GLMState` path.

Historical Phase 1 checkpoint note:

- During Phase 1, `glmax.infer.fitter` was still importable as a legacy unsupported path because `DefaultFitter()` failed through the stale `_run_default_pipeline` import.
- Final implemented end state differs: the legacy fitter modules were removed entirely in later phases of this consolidation.

## Acceptance Criteria Coverage

This phase protects:

- `glm-inference-grammar.AC1.1`: top-level grammar nouns/verbs remain stable.
- `glm-inference-grammar.AC1.2`: duplicate public-state contracts are not reintroduced.
- `glm-inference-grammar.AC3.2`: fitting still runs through the canonical grammar path while internals are being consolidated.

## Tasks

### Task 1: Add an infer-seam characterization test module

**Files:**
- Create: `tests/test_infer_seam.py`

**Implementation:**
- Add tests that document the current intended ownership boundaries:
  - `glmax.GLM` defaults to a solver from `glmax.infer.solve`.
  - public `glmax.fit`, `glmax.infer`, and `glmax.check` do not depend on `glmax.infer.fitter`, `glmax.infer.fitters`, or `glmax.infer.result`.
  - importing `glmax.infer.solve.QRSolver` remains supported.
- Add a regression test that importing or executing the canonical fit path does not touch `_run_default_pipeline`.

**Testing:**
- Favor structural and import-path assertions over brittle implementation mocking.

**Verification:**
- Run: `pytest -p no:capture tests/test_infer_seam.py`
- Expected: new seam-characterization tests pass and clearly encode the intended consolidation target.

**Commit:** `test: characterize active infer seam before consolidation`

### Task 2: Tighten existing fit API tests around stale legacy surfaces

**Files:**
- Modify: `tests/test_fit_api.py`

**Implementation:**
- Extend the existing API-surface tests so they explicitly assert:
  - `GLMState` is not publicly exported.
  - `glmax.infer` does not re-export solver or fitter internals.
  - the canonical fit route continues to operate with `glmax.infer.solve.QRSolver`.
- Add a failure-mode assertion around the stale `infer/fitter.py` path only if that path is still importable during this phase; the test should document that it is legacy and unsupported, not normalize it as public API.

**Testing:**
- Reuse existing `tests/test_fit_api.py` style and naming conventions.

**Verification:**
- Run: `pytest -p no:capture tests/test_fit_api.py tests/test_infer_seam.py`
- Expected: API-surface expectations remain green before any module deletion.

**Commit:** `test: lock grammar api against infer seam drift`

### Task 3: Record the consolidation target in the plan artifacts

**Files:**
- Modify: `docs/implementation-plans/2026-03-06-infer-seam-consolidation/phase_01.md`

**Implementation:**
- Keep this phase file updated with any verified differences discovered during implementation, especially if additional imports into stale modules appear.

**Testing:**
- N/A

**Verification:**
- Run: `rg -n "_run_default_pipeline|glmax\\.infer\\.solve|GLMState" tests/test_infer_seam.py tests/test_fit_api.py`
- Expected: characterization coverage is present in tests.

**Commit:** `docs: record infer seam characterization baseline`

## Exit Gate

Do not begin Phase 2 until the test suite clearly distinguishes:

- the supported grammar/public seam,
- the supported internal solver seam, and
- the stale fitter/result modules slated for removal.
