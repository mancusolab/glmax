# Params Disp Aux Implementation Plan

**Goal:** Align public documentation, contributor guidance, and regression coverage with the new `disp` versus `aux` terminology.

**Architecture:** Keep the package-root API stable while fixing the documentation and contributor contract around it. The current worktree has a structural docs mismatch: `mkdocs.yml`, `docs/index.md`, and `AGENTS.md` all require `docs/api/*.md`, but those files do not exist. This phase should create that missing API-doc surface and update the existing docs/tests in the same patch.

**Tech Stack:** Python 3.11, Markdown, MkDocs Material, mkdocstrings, pytest

**Scope:** 4 phases from original design (phase 4 only in this file)

**Codebase verified:** 2026-03-17 12:12:51 PDT

---

## Acceptance Criteria Coverage

This phase implements and tests:

### `params-disp-aux.AC4`: public terminology and contributor guidance align with the new contract
- **`params-disp-aux.AC4.1` Success:** README, API docs, and package docstrings describe `disp` as GLM dispersion and `aux` as family-specific.
- **`params-disp-aux.AC4.2` Success:** Negative Binomial documentation describes `alpha` as the auxiliary parameter stored in `aux`, not in `disp`.
- **`params-disp-aux.AC4.3` Failure:** stale references that describe NB `params.disp` as `alpha` are removed from docs, tests, and contributor context.
- **`params-disp-aux.AC4.4` Success:** advanced fitter/solver docs may still mention `_fit` internals, but the primary user workflow remains the top-level grammar API.

---

## Phase-by-Phase Implementation

<!-- START_SUBCOMPONENT_A (tasks 1-1) -->
<!-- START_TASK_1 -->
### Task 1: Create the missing API docs and align user-facing terminology

**Verifies:** `params-disp-aux.AC4.1`, `params-disp-aux.AC4.2`, `params-disp-aux.AC4.4`

**Files:**
- Modify: `README.md:1`
- Modify: `docs/index.md:1`
- Create: `docs/api/verbs.md`
- Create: `docs/api/nouns.md`
- Create: `docs/api/fitters.md`
- Create: `docs/api/inference.md`
- Create: `docs/api/family.md`

**Implementation:**
Create the `docs/api/` reference pages that the current `mkdocs.yml` navigation already points to, and use them to document the new terminology. `README.md` and `docs/index.md` should describe `disp` as GLM dispersion and `aux` as family-specific state. The new API pages should keep the grammar-first workflow centered on top-level nouns and verbs, reserve `_fit` details for advanced fitter/solver documentation, and explicitly document that Negative Binomial stores `alpha` in `aux` while canonical `disp` remains `1.0`.

Because the files do not exist today, do not try to move or split documentation into new directories beyond the required `docs/api/` surface.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC4.1`: public docs explain `disp` and `aux` correctly.
- `params-disp-aux.AC4.2`: Negative Binomial docs call `alpha` an auxiliary parameter.
- `params-disp-aux.AC4.4`: the docs still foreground the top-level grammar workflow.

**Verification:**
Run: `mkdocs build --strict`
Expected: MkDocs builds successfully with the new `docs/api/*.md` files and no broken nav targets.

**Commit:** `docs: add api reference pages for params aux`
<!-- END_TASK_1 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 2-2) -->
<!-- START_TASK_2 -->
### Task 2: Remove stale `params.disp == alpha` references from tests and contributor context

**Verifies:** `params-disp-aux.AC4.2`, `params-disp-aux.AC4.3`, `params-disp-aux.AC4.4`

**Files:**
- Modify: `AGENTS.md:4`
- Modify: `tests/package/test_api.py:25`
- Modify: `tests/package/test_grammar.py:37`
- Modify: `tests/data/test_glmdata.py:77`
- Modify: `tests/fit/test_fit.py:83`
- Modify: `tests/infer/test_stderr.py:41`
- Modify: `tests/glm/test_glm.py:156`

**Implementation:**
Update the contributor contract and the remaining regression assertions so nothing in the repository still describes Negative Binomial `params.disp` as `alpha`. In `AGENTS.md`, extend the `Params` carrier contract to mention `aux` and keep the docs list aligned with the files created in Task 1. In the regression suites, replace stale NB assertions with the canonical split: `params.disp` is the GLM dispersion and `params.aux` is the NB `alpha`. Include the data-layer schema assertion in `tests/data/test_glmdata.py`, which currently hard-codes the old `["beta", "disp"]` field list. Keep the public workflow and export surface unchanged while updating the semantics.

**Testing:**
Tests must verify each AC listed above:
- `params-disp-aux.AC4.2`: Negative Binomial references now point to `aux`.
- `params-disp-aux.AC4.3`: stale `params.disp == alpha` references are removed from docs, tests, and contributor guidance.
- `params-disp-aux.AC4.4`: advanced docs can still mention `_fit`, but the primary workflow remains top-level.

**Verification:**
Run: `pytest -p no:capture tests/package/test_api.py tests/package/test_grammar.py tests/fit tests/infer tests/glm tests/family tests/data`
Expected: All package, fit, infer, family, and grammar regression suites pass under the new terminology.

Run: `mkdocs build --strict`
Expected: The docs build succeeds with the updated contributor guidance and API pages.

**Commit:** `docs: align contributor contract with params aux`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_B -->
