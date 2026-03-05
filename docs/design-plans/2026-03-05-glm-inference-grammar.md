# GLM Inference Grammar Design

## Status
Draft

## Handoff Decision
- Current decision: blocked
- Ready for implementation: no
- Blocking items:
  - Pending plan review and explicit approval.

## Metadata
- Date: 2026-03-05
- Slug: glm-inference-grammar
- Artifact Directory: `docs/design-plans/artifacts/2026-03-05-glm-inference-grammar`

## Summary
This design rebuilds `glmax` around an explicit GLM-only inference grammar so users compose workflows through stable top-level nouns (`GLMData`, `Params`, `FitResult`, etc.) and verbs (`specify`, `predict`, `fit`, `infer`, `check`) instead of legacy wrapper-heavy paths. The core intent is not new math; it is contract clarity: one canonical fit entrypoint, one canonical data boundary, and one canonical parameter shape that always carries both mean coefficients and dispersion.

The implementation approach is phased and contract-first. It first standardizes public exports and shared schemas, then moves validation/canonicalization into `GLMData` and `Params`, then refactors fit/predict around those contracts, and finally layers inference/diagnostics as post-fit operations that must reuse cached fit artifacts (no refitting). The final phases remove compatibility shims and migrate tests/docs so the new grammar is enforced by both import-surface tests and full `pytest -p no:capture` regression coverage.

## Problem Statement
`glmax` currently exposes fitting/inference behavior primarily through routines and compatibility wrappers that were optimized for incremental evolution, not for a stable composable GLM-specific inference grammar. This makes it harder to compose modeling workflows across specification, fitting, inference, and diagnostics, especially when jointly handling mean parameters and dispersion/scale parameters as first-class inferential targets.

We need a GLM-limited "Grammar of Inference" with explicit nouns and verbs that are exposed at the package top level, keep JAX-first transformation-friendly contracts, and separate fitting from downstream inference/diagnostics without refitting. The redesign should replace alpha-stage legacy API surfaces directly and provide a single coherent contract that supports prediction, fitting, inference, and diagnostics with joint mean/dispersion parameter handling.

## Definition of Done
We will redesign `glmax` around a GLM-scoped Grammar of Inference where primary nouns and verbs are top-level API objects/functions, with canonical usage centered on `fit(model, data, init=None, *, fitter=...)`. `GLMData` is a first-class input object users can construct directly, joint parameter handling includes both mean parameters and dispersion/scale, and no new families are added in this phase. Since this is alpha-stage, we will replace old surfaces directly (no backward-compat contract), and the completion gate is an updated test suite passing under the new API plus migrating `README.rst` to `README.md`.

## Goals and Non-Goals
### Goals
- Expose a GLM-scoped Grammar of Inference as top-level `glmax` nouns and verbs.
- Make `fit(model, data, init=None, *, fitter=...) -> FitResult` the canonical fitting contract.
- Make `GLMData` and `Params(beta, disp)` first-class contracts for validation and joint parameter handling.
- Keep fitting, inference, prediction, and diagnostics separated so `infer`/`check` never refit.
- Complete migration with updated tests and a Markdown README (`README.md`).

### Non-Goals
- Adding new distribution families or link functions in this phase.
- Building a universal probabilistic programming framework.
- Defining full differentiation-policy semantics (deferred to a follow-on design).
- Preserving backward-compatibility wrappers or alias-heavy transition APIs.

## Existing Patterns
Confirmed patterns and constraints from codebase investigation:
- `glmax` currently re-exports families/links, `fit`, `GLM`, `GLMState`, solver/error classes, and `infer.irls` at package root (`src/glmax/__init__.py`).
- `GLMState` and `IRLSState` are duplicated across multiple modules (`src/glmax/glm.py`, `src/glmax/infer/result.py`, `src/glmax/infer/fitters.py`, `src/glmax/infer/state.py`), which creates schema drift risk.
- Dispersion (`alpha`) is already a first-class fitting concern in numerics (iterative `family.update_dispersion(...)` in `IRLSFitter`, NB-specific `estimate_dispersion(...)` bootstrap in `fit.py`).
- Inference internals are already mostly strategy-based (`fitters`, `solvers`, `inference`), while some compatibility/deprecation shims remain (`infer/optimize.py`, `infer/solve.py`, `infer/stderr.py`).

Intentional divergence in this design:
- Replace legacy array-first and wrapper surfaces directly (alpha-stage policy), instead of preserving compatibility shims.
- Move public contracts from implicit state bundles to explicit grammar nouns (`GLMData`, `Params`, `FitResult`, `InferenceResult`, `Diagnostics`) with top-level verbs (`specify`, `predict`, `fit`, `infer`, `check`).

## Model Acquisition Path
- Path: `existing-codebase-port`
- Why this path: The work is an architectural/API reorganization of an existing GLM implementation and test suite, not a new model introduction.
- User selection confirmation: Confirmed in conversation on 2026-03-05, including direct replacement (no backward compatibility) and no new family additions.

## Required Workflow States
- model_path_decided: yes
- codebase_investigation_complete_if_port: yes
- simulation_contract_complete_if_in_scope: n/a

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| SRC-1 | `src/glmax/__init__.py` | local-code | Current root-level API exports and public surface | high |
| SRC-2 | `src/glmax/fit.py` | local-code | Canonical fit workflow, boundary normalization, and NB dispersion bootstrap | high |
| SRC-3 | `src/glmax/glm.py` | local-code | Current `GLM`/`GLMState` contracts and wrapper fit behavior | high |
| SRC-4 | `src/glmax/infer/` | local-code | Fitter/solver/inference strategies plus compatibility shims and duplicate state contracts | high |
| SRC-5 | `src/glmax/family/dist.py` | local-code | Family-level weight and dispersion update hooks | high |
| SRC-6 | `tests/test_fit_api.py`, `tests/test_glm.py`, `tests/test_fitters.py` | local-tests | Existing API and compatibility assumptions to replace | high |

## Model Option Analysis (Required When `suggested-model`)
Not applicable (`existing-codebase-port`).

## Existing Codebase Port Contract (Required When `existing-codebase-port`)
- Porting objective: Replace alpha-stage public GLM fit/inference surfaces with a GLM-specific grammar API while preserving core family/link numerics and current family coverage.
- Source selection confirmation: Local repository is the source of truth for behavior and migration targets.

### Source Pin
| Source ID | Source Type (`local-directory` or `github-url`) | Path/URL | Commit/Tag | Notes |
| --- | --- | --- | --- | --- |
| PORT-SRC-1 | local-directory | `/Users/nicholas/Projects/glmax` | `0bdf4e3` | Working tree with active alpha-stage refactors and design artifacts |

### Behavior Inventory And Parity Targets
| Behavior ID | Surface (`cli`/`api`/`numerics`/`io`) | Current Behavior | Target Behavior | Evidence Plan (tests/golden outputs) |
| --- | --- | --- | --- | --- |
| PORT-BHV-1 | api | Public fit flow is split across `glmax.fit`, `GLM.fit`, and infer exports | Public grammar verbs (`specify/predict/fit/infer/check`) are top-level and unambiguous | API contract tests for import surface and signatures |
| PORT-BHV-2 | numerics | Dispersion handled via `alpha` in fit/family internals, with NB-specific bootstrap/update paths | Dispersion remains first-class in `Params.disp` and `FitResult` across supported families | Family regression tests across Gaussian/Binomial/Poisson/NegativeBinomial |
| PORT-BHV-3 | api | Wrapper compatibility behavior is explicitly tested | Compatibility wrappers removed; tests updated to grammar-native behavior | Removal/update of wrapper-parity tests + new grammar tests |
| PORT-BHV-4 | docs | Root docs centered on `README.rst` and compatibility language | Root docs centered on `README.md` and grammar API examples | Docs regression checks + README migration review |

## Codebase Investigation Findings (Required When `existing-codebase-port`)
- Investigation mode: `local-directory`
- Investigation completion: yes
- Investigator: `scientific-codebase-investigation-pass`

| Finding ID | Source Scope | Summary | Evidence (file:line or commit:path:line) | Status (`confirmed`/`discrepancy`/`addition`/`missing`) |
| --- | --- | --- | --- | --- |
| PORT-INV-1 | API exports | Root package currently exports legacy + compatibility-heavy surface (`fit`, `GLM`, `GLMState`, `irls`, solvers) | `src/glmax/__init__.py:7`, `src/glmax/__init__.py:21`, `src/glmax/__init__.py:24`, `src/glmax/__init__.py:28` | confirmed |
| PORT-INV-2 | State schemas | `GLMState`/`IRLSState` contracts are duplicated in multiple modules | `src/glmax/glm.py:21`, `src/glmax/infer/result.py:6`, `src/glmax/infer/fitters.py:15`, `src/glmax/infer/state.py:6` | confirmed |
| PORT-INV-3 | Fitting verbs | Public fitting is currently split between module fit, wrapper fit, and infer exports | `src/glmax/fit.py:117`, `src/glmax/glm.py:131`, `src/glmax/infer/optimize.py:15` | confirmed |
| PORT-INV-4 | Dispersion handling | Dispersion is already updated in iterative fitting and family-specific hooks | `src/glmax/infer/fitters.py:122`, `src/glmax/family/dist.py:82`, `src/glmax/family/dist.py:285` | confirmed |
| PORT-INV-5 | Tests/docs dependency | Existing tests and docs assert wrapper compatibility and `README.rst` assumptions | `tests/test_fit_api.py:109`, `tests/test_glm.py:195`, `README.rst:52`, `docs/index.md:31` | confirmed |
| PORT-INV-6 | Validation tooling | Design/test plans expect `pytest -p no:capture`, while hatch script still runs plain pytest | `docs/test-plans/2026-03-04-glm-fit-api.md:13`, `pyproject.toml:74` | addition |

## External Research Findings (When Triggered)
No external research required; this design is codebase-anchored.

## Mathematical Sanity Checks
- Summary: This design changes contracts and module boundaries, not family/link likelihood mathematics. Existing IRLS weighting, score, and covariance formulas remain the numerical baseline.
- Blocking issues: none identified for design readiness.
- Accepted risks: Temporary parity drift risk during schema migration from `GLMState/alpha` to `FitResult/Params.disp`; mitigated with family regression and boundary-failure tests.

Detailed artifacts:
- `docs/design-plans/artifacts/2026-03-05-glm-inference-grammar/model-symbol-table.md`
- `docs/design-plans/artifacts/2026-03-05-glm-inference-grammar/equation-to-code-map.md`

## Solver Strategy Decision
- User preference: sweeping API and architecture cleanup is acceptable in alpha stage, without adding new model families.
- Chosen strategy: retain existing solver abstractions (`AbstractLinearSolver` and concrete solvers) and keep fitter-level solver injection as the stable numerical seam.
- Why this strategy: solver contracts are already modular and tested; redesign scope is grammar API and contract clarity, not backend linear-algebra replacement.

## Solver Translation Feasibility
- Summary: High feasibility. Existing `infer/solvers.py` implementations can be reused with contract renaming and updated result schemas.
- Blocking constraints: Ensure `FitResult` carries the curvature intermediates needed by `infer`/`check` so no solver outputs are recomputed ad hoc.
- Custom-solver rationale (if chosen): not applicable in this phase.

Detailed artifact:
- `docs/design-plans/artifacts/2026-03-05-glm-inference-grammar/solver-feasibility-matrix.md`

## Layer Contracts
### Ingress
- Contract:
  - `specify(...) -> GLM` constructs static model specification (family/link/convention metadata).
  - `GLMData(...)` is the canonical data boundary for `fit`, `predict`, `infer`, and `check` contexts where observed data are required.
  - `fit(model: GLM, data: GLMData, init: Params | None = None, *, fitter: Fitter | None = None) -> FitResult`.
- Rejection rules:
  - Reject non-numeric, non-finite, rank-invalid, or shape-mismatched arrays at `GLMData` boundary.
  - Reject incompatible `model`/`fitter` contracts before entering iterative numerics.
  - Reject family/link incompatibilities before fit loop execution.

### Pipeline
- Contract:
  - Build fit-ready problem state from `GLM` and `GLMData`.
  - Execute fitter updates for joint parameter state (`beta`, `disp`) using family weight + dispersion hooks.
  - Materialize `FitResult` once, then reuse in `infer` and `check` without refitting.
- Validation-first checks:
  - `GLMData` boundary validation occurs before any iterative JAX loop.
  - `infer`/`check` validate presence and consistency of required `FitResult` fields and reject missing curvature prerequisites.

### Numerics
- Contract:
  - Preserve current family/link math and existing supported families.
  - Keep family-level hooks (`calc_weight`, `update_dispersion`) as the mechanism for dispersion-aware updates.
  - Represent joint parameters through `Params(beta, disp)` regardless of family-specific dispersion conventions.
- Result/status semantics:
  - `FitResult` reports `params`, objective/convergence metadata, and curvature blocks needed for downstream covariance/diagnostics.
  - `InferenceResult` and `Diagnostics` are derived artifacts; neither may trigger fitter execution.

### Egress
- Contract:
  - Top-level exports provide only grammar nouns/verbs plus families/links and solver/fitter contracts needed for composition.
  - Legacy wrappers/shims are removed from public surface in this alpha-stage replacement.
- Output/exit-code mapping:
  - Library API uses Python exceptions for boundary and contract violations.
  - No CLI exit-code changes are in scope.

## Data Conversion and Copy Strategy
- Source format: in-memory array-like `X`, `y`, and optional `offset/weights/mask`.
- Conversion mode: single-copy fallback at `GLMData` boundary via `jnp.asarray` normalization.
- Rationale: deterministic one-time normalization at ingress avoids repeated conversions inside iterative numerics and keeps JAX tracing behavior predictable.

## Multi-Input Reconciliation Contract (Required When Multiple Tabular Sources Feed Numerics)
Not applicable for this phase. Current scope assumes pre-aligned in-memory arrays entering `GLMData`.

## Validation Strategy
- Boundary checks:
  - Enforce `X` as rank-2 and `y` as rank-1 with shared sample dimension.
  - Enforce optional vectors (`offset`, `weights`, `mask`) as scalar-broadcastable or length-`n`.
- Shape/range/domain checks:
  - Reject non-finite values in all required numeric boundary fields.
  - Preserve family/link domain checks and reject invalid parameterization early.
- Multi-input alignment checks (key uniqueness, overlap expectations, deterministic row ordering):
  - Not applicable in this phase.
- Failure semantics:
  - Fail fast with deterministic `TypeError`/`ValueError` at boundary contracts.
  - Convergence failures remain explicit in `FitResult` metadata rather than silent retries.

## Testing and Verification Strategy
- TDD scope:
  - Grammar contracts for nouns/verbs are test-first (import surface, signatures, and boundary failures).
  - Replacement tests assert no legacy wrapper contract is required.
- Regression strategy:
  - Preserve family-level numerical regression coverage for Gaussian/Binomial/Poisson/NegativeBinomial under the new API.
  - Replace compatibility-wrapper parity tests with grammar-native end-to-end tests (`specify -> fit -> infer/check/predict`).
  - Add explicit tests that `infer`/`check` do not refit.
- Verification commands:
  - `pytest -p no:capture tests/test_fit_api.py`
  - `pytest -p no:capture tests/test_glm.py`
  - `pytest -p no:capture tests/test_fitters.py`
  - `pytest -p no:capture tests`

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Public Grammar Contract Surface
**Goal:** Define and export the stable top-level grammar nouns and verbs.

**Components:**
- Root API exports in `src/glmax/__init__.py` are rewritten around grammar nouns (`GLMData`, `Params`, `GLM`, `Fitter`, `FitResult`, `InferenceResult`, `Diagnostics`) and verbs (`specify`, `predict`, `fit`, `infer`, `check`).
- Core contract module for shared nouns and result schemas (`src/glmax/contracts.py`) to eliminate duplicate state definitions.
- `src/glmax/glm.py` becomes static model specification (`GLM`) plus `specify(...)` construction behavior.

**Dependencies:** none.

**Done when:** Grammar import surface is stable and tested, duplicate public-state contracts are removed, and `glm-inference-grammar.AC1.1`, `glm-inference-grammar.AC1.2` are satisfied.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Canonical Data And Parameter Boundaries
**Goal:** Move boundary validation and parameter canonicalization into grammar nouns.

**Components:**
- `GLMData` owns canonicalization and validation of `X`, `y`, `offset`, `weights`, and masks.
- `Params` becomes the canonical joint-parameter container with `beta` and `disp`.
- Family-facing dispersion conventions are aligned to `Params.disp` without adding new families.

**Dependencies:** Phase 1.

**Done when:** Fit/infer entrypoints consume `GLMData` + `Params` contracts, invalid boundary data fails deterministically, and `glm-inference-grammar.AC2.1`, `glm-inference-grammar.AC2.2`, `glm-inference-grammar.AC2.3` are satisfied.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Fit And Predict Verb Refactor
**Goal:** Re-anchor fitting and prediction on model/data grammar contracts.

**Components:**
- `src/glmax/fit.py` is refactored to canonical `fit(model, data, init=None, *, fitter=...) -> FitResult`.
- `predict(model, params, data)` is exposed as a pure prediction verb.
- Fit result schema includes convergence/objective metadata and curvature intermediates required by downstream inference/diagnostics.

**Dependencies:** Phase 2.

**Done when:** End-to-end fitting works through grammar contracts across supported families, dispersion is persisted in `FitResult.params.disp`, and `glm-inference-grammar.AC3.1`, `glm-inference-grammar.AC3.2`, `glm-inference-grammar.AC3.4` are satisfied.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Inference And Diagnostics Verbs
**Goal:** Provide explicit post-fit inference and diagnostics without refitting.

**Components:**
- `infer(model, fit_result, ...) -> InferenceResult` is implemented around existing covariance/testing strategies in `src/glmax/infer/inference.py`.
- Diagnostics boundary is implemented in `src/glmax/infer/diagnostics.py` and exported as top-level `check(...)`.
- `infer` and `check` consume cached fit artifacts only; no fitter calls are permitted.

**Dependencies:** Phase 3.

**Done when:** Inference and diagnostics verbs are contract-tested, non-refit guarantees are enforced in tests, and `glm-inference-grammar.AC3.3`, `glm-inference-grammar.AC5.1`, `glm-inference-grammar.AC5.3` are satisfied.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Legacy Surface Removal
**Goal:** Complete alpha-stage direct replacement by removing compatibility APIs and shims.

**Components:**
- Remove wrapper-style public behavior (`GLM.fit` compatibility semantics and stale wrapper tests).
- Retire deprecated infer re-export modules (`src/glmax/infer/optimize.py`, `src/glmax/infer/solve.py`, `src/glmax/infer/stderr.py`) or reduce them to non-public internals required by active modules only.
- Remove stale duplicate state modules once grammar contracts are canonical.

**Dependencies:** Phase 4.

**Done when:** Public API no longer exposes legacy compatibility seams, updated tests encode replacement behavior, and `glm-inference-grammar.AC4.1`, `glm-inference-grammar.AC4.2`, `glm-inference-grammar.AC4.3` are satisfied.
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Test And Documentation Migration
**Goal:** Finalize replacement with green test suite and Markdown-facing user docs.

**Components:**
- Update API and regression tests to grammar-native usage and remove obsolete compatibility assertions.
- Migrate root documentation from `README.rst` to `README.md` with grammar-first examples.
- Update docs pages referencing old wrapper semantics (`docs/index.md`, `docs/api/glm.md`) to new contracts.
- Align verification workflow to use `pytest -p no:capture` consistently.

**Dependencies:** Phase 5.

**Done when:** Updated suite passes with `pytest -p no:capture`, docs reflect grammar API, and `glm-inference-grammar.AC5.4`, `glm-inference-grammar.AC6.1`, `glm-inference-grammar.AC6.2`, `glm-inference-grammar.AC6.3` are satisfied.
<!-- END_PHASE_6 -->

## Simulation And Inference-Consistency Validation
- In scope: no
- Simulate entrypoint/signature: n/a
- Inputs: n/a
- Outputs: n/a
- Seed/RNG policy: n/a

### Assumption Alignment
| Inference Assumption | Simulation Rule | Mismatch Risk | Mitigation |
| --- | --- | --- | --- |
| n/a | n/a | n/a | n/a |

### Planned Validation Experiments
| Experiment ID | Type (recovery/SBC/PPC) | Success Criterion | Notes |
| --- | --- | --- | --- |
| SIM-1 | n/a | n/a | Out of scope for this API-architecture redesign |

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | Contract migration may temporarily break family-level parity during state-schema rewrite | high | Gate each phase with regression tests on Gaussian/Binomial/Poisson/NegativeBinomial | implementation owner |
| R2 | Removing wrappers could create abrupt breakage for internal users running old examples | medium | Ensure README/docs migration lands in same phase as API replacement and remove stale examples atomically | implementation owner |
| R3 | Dispersion semantics differ by family; forcing one `Params.disp` field may cause ambiguity | medium | Document family-specific interpretation of `disp` in glossary/API docs and enforce deterministic defaults | implementation owner |
| R4 | Deprecated shim removal may expose hidden internal imports | medium | Add targeted import-surface tests and module-level search before deletion | implementation owner |
| R5 | `fit(model, data, ...)` strictness may reduce usability for quick experimentation | low | Provide clear constructor examples for `GLMData` in README and docs API pages | implementation owner |

## Additional Considerations
- Differentiation-mode policy (`diff=...`) is intentionally deferred to a follow-on design and is not part of this acceptance scope.
- This design follows `python-module-design` constraints: boundaries are defined at cohesive module level, not file-per-concept taxonomy.

## Acceptance Criteria
### glm-inference-grammar.AC1: Top-level Grammar API Is Canonical
- **glm-inference-grammar.AC1.1 Success:** `glmax` top-level exports include `GLMData`, `Params`, `GLM`, `Fitter`, `FitResult`, `InferenceResult`, `Diagnostics`, and verbs `specify`, `predict`, `fit`, `infer`, `check`.
- **glm-inference-grammar.AC1.2 Success:** Canonical fit entrypoint is `fit(model, data, init=None, *, fitter=...)` and returns `FitResult`.
- **glm-inference-grammar.AC1.3 Failure:** `fit(...)` rejects non-`GLM` model inputs with deterministic contract errors.

### glm-inference-grammar.AC2: Data And Parameter Nouns Enforce Joint Contracts
- **glm-inference-grammar.AC2.1 Success:** `GLMData` accepts valid `(X, y)` with optional `offset`, `weights`, and mask fields.
- **glm-inference-grammar.AC2.2 Success:** `Params` always includes `beta` and `disp`; `FitResult.params` uses this schema.
- **glm-inference-grammar.AC2.3 Failure:** `GLMData` rejects non-numeric/non-finite values and shape mismatches deterministically.
- **glm-inference-grammar.AC2.4 Edge:** Families with fixed/trivial dispersion still populate `disp` with documented deterministic convention.

### glm-inference-grammar.AC3: Fit/Infer/Check Separation Preserves Joint Inference Flow
- **glm-inference-grammar.AC3.1 Success:** Gaussian, Binomial, Poisson, and NegativeBinomial fit successfully via grammar API.
- **glm-inference-grammar.AC3.2 Success:** Dispersion updates are applied through family hooks and final `disp` is stored in `FitResult`.
- **glm-inference-grammar.AC3.3 Failure:** `infer` and `check` never call fitters/refit loops.
- **glm-inference-grammar.AC3.4 Success:** `FitResult` includes convergence/objective metadata plus curvature artifacts required by `infer`.

### glm-inference-grammar.AC4: Alpha-Stage Direct Replacement Is Enforced
- **glm-inference-grammar.AC4.1 Success:** Legacy compatibility wrapper behavior (for example `GLM.fit` compatibility semantics) is removed from public API.
- **glm-inference-grammar.AC4.2 Success:** Deprecated infer shim exports are removed or made non-public.
- **glm-inference-grammar.AC4.3 Success:** Tests are rewritten for grammar-native contracts instead of wrapper parity expectations.

### glm-inference-grammar.AC5: Updated Test Suite Validates New API
- **glm-inference-grammar.AC5.1 Success:** New/updated tests cover `specify`, `predict`, `fit`, `infer`, and `check` contracts.
- **glm-inference-grammar.AC5.2 Success:** Family-level numerical regression tests pass for existing supported families (no new families added).
- **glm-inference-grammar.AC5.3 Failure:** Invalid contract usage for model/data/params/inference inputs raises deterministic boundary errors.
- **glm-inference-grammar.AC5.4 Success:** Full test suite passes with `pytest -p no:capture`.

### glm-inference-grammar.AC6: Documentation Migration To Markdown Is Complete
- **glm-inference-grammar.AC6.1 Success:** `README.md` exists with grammar-first API usage.
- **glm-inference-grammar.AC6.2 Success:** `README.rst` is removed as the root project readme surface.
- **glm-inference-grammar.AC6.3 Success:** Docs pages are updated to grammar API terminology and no longer describe wrapper-compatibility as canonical behavior.

## Glossary
- **Grammar of Inference**: The design pattern of explicit modeling nouns and verbs that defines how users specify models, fit them, and run downstream inference consistently.
- **GLM (Generalized Linear Model)**: The model family in scope for this redesign; no expansion to new families or links in this phase.
- **GLMData**: Canonical validated input container for `X`, `y`, and optional vectors like `offset`, `weights`, `mask`.
- **Params (`beta`, `disp`)**: Joint parameter container for mean coefficients (`beta`) and dispersion/scale (`disp`) across families.
- **Dispersion (`disp` / historical `alpha`)**: Non-mean parameter controlling variance scale; kept first-class and propagated through fit outputs.
- **IRLS (Iteratively Reweighted Least Squares)**: Existing iterative fitting strategy that remains part of the numerical core.
- **Fitter**: Strategy object controlling optimization updates for model parameters during `fit`.
- **FitResult**: Primary fit artifact carrying final parameters, convergence/objective metadata, and intermediates needed later by inference/diagnostics.
- **InferenceResult**: Post-fit statistical output (for example, standard errors/tests) derived from `FitResult` without running fitting loops.
- **Diagnostics (`check`)**: Post-fit checks derived from model + fit artifacts, explicitly separated from fitting.
- **Curvature artifacts**: Stored second-order information from fitting required for covariance and inference computations.
- **Boundary validation**: Deterministic shape/type/domain checks at API ingress that fail fast with explicit errors.
- **Compatibility shims/wrappers**: Legacy adapter surfaces being removed in this alpha-stage direct replacement.
- **JAX-first contract**: API/data handling designed to preserve JAX-friendly tracing/transformation behavior.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-03-05 | N/A | Draft | Plan created | |
