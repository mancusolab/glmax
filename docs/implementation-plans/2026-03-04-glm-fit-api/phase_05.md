# GLM Fit API Implementation Plan

**Goal:** Update docs and migration guidance to promote `gx.fit(...)` and complete repository verification.

**Architecture:** Publish one canonical user path (`import glmax as gx; gx.fit(...)`) and document `GLM.fit` as compatibility while preserving required verification commands.

**Tech Stack:** MkDocs, Python 3.11, pytest.

**Scope:** 5 phases from original design (this file covers Phase 5).

**Codebase verified:** 2026-03-04 (America/Los_Angeles)

---

## Acceptance Criteria Coverage

This phase implements and tests:

### glm-fit-api.AC5: Documentation And Verification Are Updated
- **glm-fit-api.AC5.1 Success:** Documentation shows `gx.fit(...)` as the recommended usage pattern.
- **glm-fit-api.AC5.2 Success:** Migration guidance documents wrapper compatibility and future deprecation direction.
- **glm-fit-api.AC5.3 Success:** Verification commands `pytest -p no:capture tests/test_glm.py` and `pytest -p no:capture tests` pass at completion.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: Add user-facing API docs for `gx.fit`

**Verifies:** glm-fit-api.AC5.1

**Files:**
- Create: `docs/api/glm.md`
- Modify: `docs/index.md:1-120`
- Verify: `mkdocs.yml` nav already references `api/glm.md` (no edit required)

**Implementation:**
- Create API page content showing `gx.fit(...)` as preferred usage.
- Update docs landing page examples from legacy style to package-level API.
- Ensure MkDocs nav points only to existing docs files.

**Testing:**
- Build docs and verify navigation resolves.

**Verification:**
- Run: `mkdocs build`
- Expected: build succeeds without missing-file nav errors.

**Commit:** `docs: publish gx.fit preferred usage docs`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Publish migration guidance for `GLM.fit` callers

**Verifies:** glm-fit-api.AC5.2

**Files:**
- Modify: `README.rst:40-140`
- Modify: `docs/api/glm.md`

**Implementation:**
- Add migration mapping from `GLM(...).fit(...)` to `gx.fit(model, X, y, ...)`.
- Document current compatibility guarantees and deprecation-ready direction.

**Testing:**
- Validate code snippets and docs links.

**Verification:**
- Run: `mkdocs build`
- Expected: migration content renders and links correctly.

**Commit:** `docs: add GLM.fit migration guidance`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Run final repository verification commands

**Verifies:** glm-fit-api.AC5.3

**Files:**
- Modify: `docs/implementation-plans/2026-03-04-glm-fit-api/phase_05.md` (completion notes section)

**Implementation:**
- Run required test commands and record pass state in completion notes.

**Testing:**
- Verification commands are the deliverable for this task.

**Verification:**
- Run: `pytest -p no:capture tests/test_glm.py`
- Expected: pass.
- Run: `pytest -p no:capture tests`
- Expected: pass.

**Commit:** `chore: finalize verification evidence for gx.fit migration`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

## Completion Notes

- `mkdocs build`: passed.
- `pytest -p no:capture tests/test_glm.py`: passed (12 tests).
- `pytest -p no:capture tests`: passed (56 tests).
