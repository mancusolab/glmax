# GLM Fit API And Inference Reorganization Design

## Status
Draft

## Handoff Decision
- Current decision: blocked
- Ready for implementation: no
- Blocking items:
  - Pending plan review and explicit approval.

## Metadata
- Date: 2026-03-04
- Slug: glm-fit-api
- Artifact Directory: `docs/design-plans/artifacts/2026-03-04-glm-fit-api`

## Summary
This design plan moves GLM fitting orchestration from `GLM.fit(...)` to a package-level `gx.fit(model, X, y, offset=None, *, fitter=..., solver=..., covariance=..., tests=...)` entrypoint. The goal is to make user-priority inputs first-class, keep `GLM` focused on static model specification, and centralize runtime inference behavior (IRLS execution, linear solver selection, covariance estimation, and hypothesis testing) under a cohesive inference boundary.

The implementation is phased to preserve numerical behavior and migration safety: introduce the new API and contracts, reorganize model/inference modules, route execution through `gx.fit`, keep `GLM.fit(...)` as a compatibility wrapper, then update docs and verify with `pytest -p no:capture`. Acceptance criteria require family-level numerical parity (Gaussian/Binomial/Poisson/NegativeBinomial), preserved convergence metadata, validation-first failures, and output equivalence between old and new entrypoints.

## Problem Statement
The current public API centers `GLM.fit(...)` inside the model object (`src/glmax/glm.py`) while orchestration logic spans optimizer, solver, and standard-error modules (`src/glmax/infer/optimize.py`, `src/glmax/infer/solve.py`, `src/glmax/infer/stderr.py`). This makes it harder to compare and evolve fitting strategies at one stable entrypoint and mixes static model specification with dynamic fitting behavior.

We need a high-level `gx.fit(model, X, y, ...)` API where user-priority inputs are first-class, while still supporting modular inference components (algorithm, linear solve backend, covariance estimator, hypothesis tests). The refactor should reorganize modules around stable architectural seams and avoid unnecessary micro-file fragmentation.

## Definition of Done
- Implement `gx.fit(model, X, y, offset=None, *, fitter=..., solver=..., covariance=..., tests=...)` as the primary high-level API.
- Execute a larger refactor/reorg of files now, centered on static model spec vs inference runtime, with fitters/solvers co-located under one inference boundary.
- Preserve incremental delivery quality (tests and compatibility strategy), but a full rewrite is not required.
- Produce a design plan under slug `glm-fit-api` that specifies module layout, migration steps, and acceptance criteria for both API and reorg.

## Goals and Non-Goals
### Goals
- Make `gx.fit(...)` the orchestration entrypoint with user-priority argument ordering.
- Separate static model specification from runtime fit state and inference orchestration.
- Consolidate inference strategy components in one cohesive inference seam.
- Preserve current statistical behavior and test parity for Gaussian/Binomial/Poisson/NegativeBinomial fits.

### Non-Goals
- Replacing IRLS with a new optimization family in this refactor.
- Changing distribution mathematics or link-function formulas.
- Introducing strict public API stability guarantees in this phase.

## Existing Patterns
Confirmed existing patterns:
- `GLM` currently bundles fit orchestration and produces `GLMState` (`src/glmax/glm.py:16`, `src/glmax/glm.py:54`, `src/glmax/glm.py:130`).
- IRLS is already an independent algorithm function (`src/glmax/infer/optimize.py:18`).
- Linear solver backends already share one abstraction (`src/glmax/infer/solve.py:14`, `src/glmax/infer/solve.py:74`).
- Standard-error estimators already share one abstraction (`src/glmax/infer/stderr.py:13`, `src/glmax/infer/stderr.py:70`).
- Family/link abstractions are separated from inference internals (`src/glmax/family/dist.py:18`).
- Test baseline compares outputs against statsmodels across key families (`tests/test_glm.py:84`, `tests/test_glm.py:149`).

Intentional divergence:
- Move high-level orchestration from `GLM.fit(...)` to package-level `gx.fit(...)`.
- Reorganize modules to center the model-vs-inference boundary while keeping fitters and solvers together.

## Model Acquisition Path
- Path: `existing-codebase-port`
- Why this path: Refactor and API reorganization are anchored to a pre-existing GLM implementation in this repository.
- User selection confirmation: confirmed in conversation.

## Required Workflow States
- model_path_decided: yes
- codebase_investigation_complete_if_port: yes
- simulation_contract_complete_if_in_scope: n/a

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| SRC-1 | `src/glmax/glm.py` | local-code | Current public API and model object behavior | high |
| SRC-2 | `src/glmax/infer/` | local-code | Existing inference seams (optimize/solve/stderr) | high |
| SRC-3 | `src/glmax/family/` | local-code | Family and link contracts used by IRLS | high |
| SRC-4 | `tests/test_glm.py` | local-code | Baseline statistical behavior and parity checks | high |

## Model Option Analysis (Required When `suggested-model`)
Not applicable (`existing-codebase-port`).

## Existing Codebase Port Contract (Required When `existing-codebase-port`)
- Porting objective: Reorganize API and module boundaries without losing current statistical behavior.
- Source selection confirmation: local repository source of truth.

### Source Pin
| Source ID | Source Type (`local-directory` or `github-url`) | Path/URL | Commit/Tag | Notes |
| --- | --- | --- | --- | --- |
| PORT-SRC-1 | local-directory | `/Users/nicholas/Projects/glmax` | current working tree | In-progress implementation baseline |

### Behavior Inventory And Parity Targets
| Behavior ID | Surface (`cli`/`api`/`numerics`/`io`) | Current Behavior | Target Behavior | Evidence Plan (tests/golden outputs) |
| --- | --- | --- | --- | --- |
| PORT-BHV-1 | api | Users call `GLM(...).fit(X, y, ...)` | Users call `gx.fit(model, X, y, ...)` (primary), with compatibility wrapper | API tests + compatibility tests |
| PORT-BHV-2 | numerics | IRLS + solver + SE produce family-specific estimates | Same numerical behavior and convergence semantics | Existing tests + added cross-entrypoint parity tests |
| PORT-BHV-3 | api | `GLMState` includes params, SE, p-values, diagnostics | Result object remains functionally equivalent with clearer separation of fit-state/result-state | result field parity assertions |
| PORT-BHV-4 | api | Family/link behavior wired through `ExponentialFamily` | Preserve family/link semantics and allowed links | Existing family tests and regression checks |

## Codebase Investigation Findings (Required When `existing-codebase-port`)
- Investigation mode: `local-directory`
- Investigation completion: yes
- Investigator: `scientific-codebase-investigation-pass`

| Finding ID | Source Scope | Summary | Evidence (file:line or commit:path:line) | Status (`confirmed`/`discrepancy`/`addition`/`missing`) |
| --- | --- | --- | --- | --- |
| PORT-INV-1 | API | `GLM.fit` is current high-level entrypoint | `src/glmax/glm.py:130` | confirmed |
| PORT-INV-2 | Inference | IRLS already separated from model class | `src/glmax/infer/optimize.py:18` | confirmed |
| PORT-INV-3 | Inference | Solver strategy abstraction exists and is extensible | `src/glmax/infer/solve.py:14` | confirmed |
| PORT-INV-4 | Inference | Covariance/SE abstraction exists and is extensible | `src/glmax/infer/stderr.py:13` | confirmed |
| PORT-INV-5 | Families | Family hierarchy and link integration already stable | `src/glmax/family/dist.py:18` | confirmed |
| PORT-INV-6 | Tests | Statsmodels parity tests exist for four families | `tests/test_glm.py:84` | confirmed |

## External Research Findings (When Triggered)
No external research required for this refactor design.

## Mathematical Sanity Checks
- Summary: Mathematical formulas for existing families, links, IRLS updates, and standard-error estimators remain unchanged by this design.
- Blocking issues: none identified.
- Accepted risks: temporary migration risk if wrapper and new entrypoint diverge.

Detailed artifacts:
- `docs/design-plans/artifacts/2026-03-04-glm-fit-api/model-symbol-table.md`
- `docs/design-plans/artifacts/2026-03-04-glm-fit-api/equation-to-code-map.md`

## Solver Strategy Decision
- User preference: keep high-level fit API and diff-friendly fit orchestration.
- Chosen strategy: maintain pluggable linear solvers within one inference seam.
- Why this strategy: current solver abstractions are already stable; co-locating solver/fitter under inference minimizes fragmentation and supports algorithm/backend comparisons.

## Solver Translation Feasibility
- Summary: Existing `AbstractLinearSolver` contract can be reused with namespace relocation and minimal callsite changes.
- Blocking constraints: none.
- Custom-solver rationale (if chosen): n/a.

Detailed artifact:
- `docs/design-plans/artifacts/2026-03-04-glm-fit-api/solver-feasibility-matrix.md`

## Layer Contracts
### Ingress
- Contract: `gx.fit(model, X, y, offset=None, *, fitter=None, solver=None, covariance=None, tests=None, init=None, options=None)`
- Rejection rules:
  - reject invalid shape relationships between `X`, `y`, and `offset`
  - reject unsupported fitter/solver/covariance objects that fail protocol checks

### Pipeline
- Contract:
  - resolve defaults from model + fit config
  - construct runtime fit problem
  - execute fitter (IRLS) with selected linear solver
  - compute covariance and optional hypothesis tests
  - return unified result object
- Validation-first checks:
  - shape and dtype checks before any iterative loop
  - family/link compatibility checks before fit execution

### Numerics
- Contract:
  - re-use existing family/link/IRLS equations
  - preserve convergence metadata and dispersion handling
- Result/status semantics:
  - expose convergence status and iteration counts
  - preserve comparable coefficient/SE/p-value outputs against current baseline

### Egress
- Contract:
  - return result with params, covariance-derived SE, diagnostics, and prediction/test methods
  - compatibility wrapper `GLM.fit(...)` delegates to `gx.fit(...)`
- Output/exit-code mapping: Python API raises typed exceptions for invalid configuration; no CLI exit codes in scope.

## Data Conversion and Copy Strategy
- Primary numeric inputs are already array-like; convert once at ingress and preserve device-compatible arrays through fit.
- Avoid repeated materialization within the iterative loop.
- Keep conversion boundary at `gx.fit(...)` ingress.

## Multi-Input Reconciliation Contract (Required When Multiple Tabular Sources Feed Numerics)
Not applicable for current direct-array API scope.

## Validation Strategy
- Boundary checks:
  - enforce `X.shape[0] == y.shape[0]`
  - enforce offset length matches sample dimension when provided
- Shape/range/domain checks:
  - reject non-finite arrays for required numeric inputs
  - preserve family-specific domain checks through existing link/family behavior
- Multi-input alignment checks (key uniqueness, overlap expectations, deterministic row ordering): n/a
- Failure semantics:
  - input-contract failures raise early exceptions
  - convergence failures surface explicit status in result

## Testing and Verification Strategy
- TDD scope:
  - add tests for new `gx.fit(...)` entrypoint first
  - add compatibility tests that `GLM.fit(...)` delegates to new path
- Regression strategy:
  - keep existing statsmodels parity tests for all supported families
  - add entrypoint parity tests (old vs new API) on same synthetic data
- Verification commands:
  - `pytest -p no:capture tests/test_glm.py`
  - `pytest -p no:capture tests`

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Introduce API Surface And Contracts
**Goal:** Establish `gx.fit(...)` public entrypoint and explicit inference contracts without changing core math.

**Components:**
- Public API entry module exposing `fit(model, X, y, offset=None, *, fitter, solver, covariance, tests, init, options)`.
- Cohesive inference contract definitions for fitter, solver, covariance, and fit-state/result-state interfaces.
- Re-export updates in `src/glmax/__init__.py` for ergonomic top-level imports.

**Dependencies:** none.

**Done when:**
- `gx.fit(...)` is importable and validates core input contract.
- tests exist and pass for entrypoint availability and basic validation behavior.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Reorganize Model And Inference Boundaries
**Goal:** Move from model-owned orchestration to model-spec plus inference runtime organization.

**Components:**
- `GLM` becomes static model specification surface (family/link/default strategy metadata only).
- Inference boundary groups algorithm + linear solver + covariance modules in one seam.
- Result object split between dynamic runtime state and user-facing fit result representation.

**Dependencies:** Phase 1.

**Done when:**
- codebase structure reflects model-vs-inference separation.
- imports are updated with no broken public entrypoint.
- migration tests pass for existing functionality.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Route Numerical Execution Through `gx.fit`
**Goal:** Ensure IRLS/solver/covariance execution is orchestrated from `gx.fit(...)` while preserving behavior.

**Components:**
- `gx.fit(...)` execution pipeline wiring existing IRLS algorithm and solver abstractions.
- covariance estimator integration through unified strategy interface.
- hypothesis-test hook integration (initially Wald parity, extensible for Score/LRT).

**Dependencies:** Phase 2.

**Done when:**
- coefficients/SE/p-values from `gx.fit(...)` match pre-refactor behavior within tolerance.
- convergence metadata is preserved.
- tests covering api + numerical parity pass.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Compatibility Wrapper And Migration Safety
**Goal:** Preserve user continuity while making `gx.fit(...)` the primary path.

**Components:**
- `GLM.fit(...)` compatibility wrapper delegating to `gx.fit(...)`.
- deprecation-ready messaging path (if enabled later) without breaking current tests.
- explicit parity tests comparing wrapper and direct API outputs.

**Dependencies:** Phase 3.

**Done when:**
- legacy entrypoint continues to function through delegation.
- wrapper and direct API produce equivalent results on shared fixtures.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Documentation, Examples, And Final Verification
**Goal:** Publish the new API shape and confirm repository-level correctness.

**Components:**
- API docs updates showing `gx.fit(model, X, y, ...)` as preferred usage.
- migration note for users currently calling `GLM.fit(...)`.
- final verification run across full test suite.

**Dependencies:** Phase 4.

**Done when:**
- documentation reflects new preferred entrypoint.
- all relevant tests pass using required command form.
- no unresolved regressions remain.
<!-- END_PHASE_5 -->

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
| SIM-1 | n/a | n/a | n/a |

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | Import churn during module reorg can create temporary circular dependencies | medium | Keep phased migration with small commits and import checks each phase | implementation |
| R2 | Wrapper and direct API could drift in behavior over time | medium | Add parity tests that compare both entrypoints for all families | implementation |
| R3 | Result-state reshaping may break downstream users expecting exact field names | medium | Preserve aliases or compatibility accessors during migration window | implementation |

## Additional Considerations
- Module granularity policy: keep inference pieces in one cohesive boundary; do not create one-file-per-concept fragments unless stable public contracts or size pressure justify it.
- Future extensibility: this structure should support adding alternate fit algorithms without reshaping model-spec modules.

## Acceptance Criteria
### glm-fit-api.AC1: High-Level Fit API Exists And Is Preferred
- **glm-fit-api.AC1.1 Success:** Users can call `gx.fit(model, X, y)` with required arguments and receive a fit result.
- **glm-fit-api.AC1.2 Success:** Users can pass `offset` as an optional positional/keyword data argument without changing strategy defaults.
- **glm-fit-api.AC1.3 Failure:** Invalid core input shapes (`X`, `y`, `offset`) fail fast with explicit error messages.

### glm-fit-api.AC2: Inference Components Are Modular Under One Boundary
- **glm-fit-api.AC2.1 Success:** Fit algorithm, linear solver, and covariance strategy are independently swappable through `gx.fit(..., fitter=..., solver=..., covariance=...)`.
- **glm-fit-api.AC2.2 Success:** The refactor places these strategy components under one cohesive inference seam (not fragmented across unrelated taxonomy modules).
- **glm-fit-api.AC2.3 Failure:** Unsupported strategy objects are rejected by contract checks before fit execution.

### glm-fit-api.AC3: Numerical Behavior Is Preserved
- **glm-fit-api.AC3.1 Success:** For Gaussian, Binomial, Poisson, and NegativeBinomial models, `gx.fit(...)` produces coefficients consistent with existing baselines within test tolerances.
- **glm-fit-api.AC3.2 Success:** Standard errors and p-values remain consistent with existing baseline tests.
- **glm-fit-api.AC3.3 Edge:** Convergence metadata (iteration count and converged flag) remains available and semantically equivalent.

### glm-fit-api.AC4: Backward-Compatible Migration Path
- **glm-fit-api.AC4.1 Success:** `GLM.fit(...)` remains callable and delegates to `gx.fit(...)`.
- **glm-fit-api.AC4.2 Success:** Direct and wrapped entrypoints produce equivalent outputs on shared test inputs.
- **glm-fit-api.AC4.3 Failure:** If delegation cannot be completed due to invalid model/runtime configuration, the raised error is consistent between entrypoints.

### glm-fit-api.AC5: Documentation And Verification Are Updated
- **glm-fit-api.AC5.1 Success:** Documentation shows `gx.fit(...)` as the recommended usage pattern.
- **glm-fit-api.AC5.2 Success:** Migration guidance documents wrapper compatibility and future deprecation direction.
- **glm-fit-api.AC5.3 Success:** Verification commands `pytest -p no:capture tests/test_glm.py` and `pytest -p no:capture tests` pass at completion.

## Glossary
- **`gx.fit(...)`**: Proposed primary high-level orchestration API for fitting models with explicit strategy hooks.
- **`GLM.fit(...)`**: Existing model-method entrypoint that becomes a compatibility wrapper delegating to `gx.fit(...)`.
- **`GLM`**: Model surface intended to hold static specification (family/link/default strategy metadata), not runtime orchestration.
- **IRLS**: Existing iterative fitting algorithm reused in the refactor without changing core mathematics.
- **`AbstractLinearSolver`**: Solver abstraction used for pluggable linear solve backends (for example Cholesky/QR/CG).
- **Covariance estimator**: Strategy component that computes covariance/standard errors from fit outputs.
- **Inference seam**: Cohesive module boundary grouping fitter, solver, covariance, and test strategies.
- **Fit-state / result-state**: Separation between runtime execution state and user-facing fit result representation.
- **`ExponentialFamily`**: Family/link contract layer reused unchanged by the reorganization.
- **Statsmodels parity tests**: Existing regression baseline checks used to verify statistical behavior is preserved.
- **Convergence metadata**: Iteration count and convergence status that remain available and semantically equivalent.
- **Compatibility wrapper**: Backward-compatible delegation path from legacy `GLM.fit(...)` to new `gx.fit(...)`.
- **Validation-first checks**: Shape/dtype/family-link checks performed before iterative execution.
- **`existing-codebase-port`**: Selected model acquisition path indicating this design refactors existing implementation.
- **Acceptance Criteria (`glm-fit-api.AC1`-`AC5`)**: Scoped, testable success/failure conditions for API, architecture, parity, migration, and verification.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-03-04 | N/A | Draft | Plan created | codex |
