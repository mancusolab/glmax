# GLMAX Fit Pipeline Redesign Design

## Status
Draft

## Handoff Decision
- Current decision: blocked
- Ready for implementation: no
- Blocking items:
  - Pending plan review and explicit approval.

## Metadata
- Date: 2026-03-04
- Slug: glmax-fit-pipeline-redesign
- Artifact Directory: `docs/design-plans/artifacts/2026-03-04-glmax-fit-pipeline-redesign`

## Summary
This design consolidates the alpha-stage `glmax` fit/inference stack into a single canonical workflow centered on `glmax.fit`, removes duplicated orchestration logic, and replaces the current over-fragmented `infer` layout with cohesive goal-based modules. IRLS is formalized as a fitter strategy object rather than a loose optimize utility. The plan also includes targeted JAX cleanup at API boundaries, especially replacing NumPy dtype probing with JAX-native `jnp.issubdtype` checks and keeping boundary validation separate from numerics kernels. Public API behavior remains explicit with compatibility shims, and public numerics entrypoints receive standardized raw docstrings using `jax-project-engineering` section contracts.

## Problem Statement
`glmax` is still alpha-stage, but the current fit/inference internals are split across many small modules with overlapping responsibilities. Core behavior is duplicated across `glmax.fit`, `GLM.fit`, and fitter/test/covariance wrappers, making refactors risky and increasing API drift risk.

The codebase also mixes boundary normalization concerns with numerics execution concerns, and there are avoidable NumPy/JAX consistency issues (for example boundary dtype checks using NumPy probing when JAX-native checks are sufficient). This redesign is needed to make the fitting stack easier to reason about, reduce redundant abstractions, and establish one coherent fit pipeline before broader feature growth.

## Definition of Done
- `src/glmax` is reorganized around one canonical fit workflow with cohesive inference boundaries, replacing redundant wrapper/orchestration seams.
- `infer` is consolidated by goal/function (instead of many small modules), with IRLS represented as a fitter strategy object boundary.
- Duplicated behavior (fit orchestration, Wald/p-value path, state/test/covariance plumbing) is unified with explicit contracts and deprecation-safe compatibility behavior.
- JAX cleanup is completed for boundary dtype handling, including replacing NumPy dtype-discovery checks with JAX-native subdtype checks where appropriate.
- Public numerics entrypoints and compatibility APIs use raw docstrings with `jax-project-engineering` section labels (`**Arguments:**`, `**Returns:**`, and `**Raises:**` or `**Failure Modes:**`).

## Goals and Non-Goals
### Goals
- Establish one canonical public fit implementation path and reduce compatibility behavior to thin mapping-only shims.
- Replace the current `infer` micro-module split with cohesive modules aligned to workflow boundaries.
- Represent IRLS as a fitter object contract that composes solver/covariance/hypothesis strategies.
- Standardize JAX-first boundary validation logic (`jnp.asarray`, `jnp.issubdtype`, finite checks) and eliminate avoidable NumPy dtype probing.
- Improve public API documentation quality and consistency according to project docstring standards.

### Non-Goals
- Changing GLM statistical model semantics or introducing new estimation algorithms.
- Adding new model families or new CLI surfaces.
- Performing deep solver family migration (for example replacing Lineax with a different linear algebra backend).
- Introducing broad package-level subpackage restructuring outside `infer` consolidation.

## Existing Patterns
- `glmax.fit` is already documented in README as the preferred public entrypoint, while `GLM.fit` acts as compatibility wrapper (`README.rst`, `src/glmax/glm.py`, `src/glmax/fit.py`).
- Fit API tests already enforce parity between direct and wrapper entrypoints (`tests/test_fit_api.py`).
- `infer` responsibilities are currently split into many small modules (`fitter.py`, `optimize.py`, `result.py`, `state.py`, `stderr.py`, `tests.py`, `solve.py`), producing low-signal boundaries.
- Solver strategies currently use Lineax `linear_solve`; covariance and hypothesis-test strategies are already abstracted.
- Boundary validation currently uses mixed NumPy/JAX handling; this redesign preserves validation-first behavior but shifts to JAX-first dtype checks where feasible.

## Model Acquisition Path
- Path: `existing-codebase-port`
- Why this path: This effort is a structural redesign and consolidation of an existing production codebase, not a new model introduction.
- User selection confirmation: confirmed in this planning conversation on 2026-03-04.

## Required Workflow States
- model_path_decided: yes
- codebase_investigation_complete_if_port: yes
- simulation_contract_complete_if_in_scope: n/a

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| SRC-1 | `src/glmax/fit.py` | codebase | Canonical fit pipeline baseline and validation behavior | high |
| SRC-2 | `src/glmax/glm.py` | codebase | Compatibility wrapper behavior and current duplication seams | high |
| SRC-3 | `src/glmax/infer/` | codebase | Current module fragmentation and strategy contracts | high |
| SRC-4 | `tests/test_fit_api.py` | tests | Existing parity and compatibility constraints | high |

## Model Option Analysis (Required When `suggested-model`)
Not applicable. Path is `existing-codebase-port`.

## Existing Codebase Port Contract (Required When `existing-codebase-port`)
- Porting objective: Preserve GLM fit behavior while restructuring internals to remove redundant abstractions and enforce one canonical workflow.
- Source selection confirmation: Local `glmax` repository state as of `42ef67a`.

### Source Pin
| Source ID | Source Type (`local-directory` or `github-url`) | Path/URL | Commit/Tag | Notes |
| --- | --- | --- | --- | --- |
| PORT-SRC-1 | local-directory | `/Users/nicholas/Projects/glmax` | `42ef67a` | Working tree includes unrelated untracked files not part of this design |

### Behavior Inventory And Parity Targets
| Behavior ID | Surface (`cli`/`api`/`numerics`/`io`) | Current Behavior | Target Behavior | Evidence Plan (tests/golden outputs) |
| --- | --- | --- | --- | --- |
| PORT-BHV-1 | api | `glmax.fit` and `GLM.fit` return parity-tested `GLMState` outputs | Preserve parity while making `glmax.fit` the only implementation path | Existing parity tests + added delegation invariants |
| PORT-BHV-2 | numerics | IRLS loop in `infer/optimize.py` with strategy composition | Preserve numerical behavior with IRLS expressed as fitter object | Regression vs existing test baselines |
| PORT-BHV-3 | api | Type/shape/finiteness errors raised at fit boundary | Preserve failure contract with clearer JAX-first dtype checks | Boundary-failure regression tests |
| PORT-BHV-4 | api | Wald p-value behavior split across wrapper and strategy class | Single canonical p-value strategy path | Parity tests on Gaussian/non-Gaussian p-values |

## Codebase Investigation Findings (Required When `existing-codebase-port`)
- Investigation mode: `local-directory`
- Investigation completion: yes
- Investigator: `scientific-codebase-investigation-pass`

| Finding ID | Source Scope | Summary | Evidence (file:line or commit:path:line) | Status (`confirmed`/`discrepancy`/`addition`/`missing`) |
| --- | --- | --- | --- | --- |
| PORT-INV-1 | Fit APIs | Fit logic duplicated across `fit()` and `GLM.fit()` orchestration seams | `src/glmax/fit.py:24`, `src/glmax/fit.py:98`, `src/glmax/glm.py:75` | confirmed |
| PORT-INV-2 | Inference structure | `infer` has multiple tiny modules carrying related contracts | `src/glmax/infer/result.py:1`, `src/glmax/infer/state.py:1`, `src/glmax/infer/tests.py:1`, `src/glmax/infer/stderr.py:1` | confirmed |
| PORT-INV-3 | IRLS placement | IRLS currently implemented in standalone optimize utility | `src/glmax/infer/optimize.py:12` | confirmed |
| PORT-INV-4 | Boundary dtype handling | Boundary conversion checks numeric dtype via NumPy probing | `src/glmax/fit.py:17-21`, `src/glmax/glm.py:127-138` | confirmed |
| PORT-INV-5 | Statistical test duplication | Wald logic duplicated in `GLM.wald_test` and `infer.WaldTest` | `src/glmax/glm.py:164`, `src/glmax/infer/tests.py:18` | confirmed |

## External Research Findings (When Triggered)
Not triggered for this design. Changes are codebase-structure and internal contract focused.

## Mathematical Sanity Checks
- Summary: No changes to model family likelihood definitions, solver equations, or p-value formula semantics are planned; redesign is structural and contract-focused.
- Blocking issues: None identified for architecture planning.
- Accepted risks: Numerical parity depends on preserving IRLS update order and stopping logic during fitter-object migration.

Detailed artifacts:
- `docs/design-plans/artifacts/2026-03-04-glmax-fit-pipeline-redesign/model-symbol-table.md`
- `docs/design-plans/artifacts/2026-03-04-glmax-fit-pipeline-redesign/equation-to-code-map.md`

## Solver Strategy Decision
- User preference: Keep sweeping internal cleanup while preserving core GLM behavior.
- Chosen strategy: Retain Lineax-based linear solver strategy objects and reuse existing solver classes.
- Why this strategy: Existing solver behavior is already validated in test suite; current refactor scope targets orchestration and module design rather than linear algebra replacement.

## Solver Translation Feasibility
- Summary: High feasibility. Solver interfaces already exist and can be migrated into cohesive module boundaries without changing algorithmic contract.
- Blocking constraints: Must preserve input/output shape contracts and current error semantics.
- Custom-solver rationale (if chosen): Not chosen.

Detailed artifact:
- `docs/design-plans/artifacts/2026-03-04-glmax-fit-pipeline-redesign/solver-feasibility-matrix.md`

## Layer Contracts
### Ingress
- Contract: `glmax.fit(model, X, y, offset, ..., options)` remains canonical ingress for user-facing validation and normalization.
- Rejection rules: Raise built-in exceptions for non-numeric dtypes, wrong rank, shape mismatch, and non-finite values.

### Pipeline
- Contract: Canonical fit pipeline maps validated arrays + strategy objects into fitter execution and post-fit inference outputs.
- Validation-first checks: All Python exceptions for user input remain at boundary; numerics kernels avoid Python `raise` inside traced control flow.

### Numerics
- Contract: Fitter object (default IRLS fitter) receives stable arrays, family, and strategy components and returns stable state containers.
- Result/status semantics: Preserve current exception-first boundary contract and explicit convergence metadata (`num_iters`, `converged`) in `GLMState`.

### Egress
- Contract: Return `GLMState` with stable fields and strategy-consistent p-values/standard errors.
- Output/exit-code mapping: Library API only; no CLI contract changes in scope.

## Data Conversion and Copy Strategy
- Input sources are in-memory array-like inputs only.
- Conversion policy: boundary conversion to JAX arrays with JAX-first dtype validation.
- Copy behavior: single-copy fallback via `jnp.asarray` at boundary as needed; no additional tabular adapters in scope.

## Multi-Input Reconciliation Contract (Required When Multiple Tabular Sources Feed Numerics)
Not applicable for this redesign scope.

## Validation Strategy
- Boundary checks: centralize type/shape/finite checks in canonical fit boundary.
- Shape/range/domain checks: preserve current constraints for `X`, `y`, `offset`, and family/link compatibility.
- Multi-input alignment checks (key uniqueness, overlap expectations, deterministic row ordering): not applicable.
- Failure semantics: built-in exceptions at boundary only; no new exception class hierarchy.

## Testing and Verification Strategy
- TDD scope: required for all behavior changes in fit pipeline consolidation and infer-module migration.
- Regression strategy: preserve and expand parity tests between `glmax.fit` and `GLM.fit`, strategy injection tests, and boundary-failure tests.
- Verification commands:
  - `pytest -p no:capture tests/test_fit_api.py`
  - `pytest -p no:capture tests/test_glm.py`
  - `pytest -p no:capture tests`

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Canonical Fit Contract Freeze
**Goal:** Establish one explicit fit contract and remove orchestration ambiguity before structural migration.

**Components:**
- Canonical fit boundary in `src/glmax/fit.py` remains the only implementation path for orchestration and validation.
- Compatibility shim in `src/glmax/glm.py` is reduced to argument translation/deprecation behavior only.
- Public export surfaces in `src/glmax/__init__.py` and `src/glmax/infer/__init__.py` are updated to match canonical contract.

**Dependencies:** None.

**Done when:** `glmax.fit` and `GLM.fit` parity tests pass unchanged, and implementation ownership of orchestration is singular (covers `glmax-fit-pipeline-redesign.AC1.1`, `glmax-fit-pipeline-redesign.AC1.2`).
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Infer Module Consolidation
**Goal:** Replace over-fragmented infer micro-modules with cohesive goal-based module boundaries.

**Components:**
- Consolidate state/result contracts into one stable contract module (for example `src/glmax/infer/contracts.py`).
- Consolidate solver strategy interfaces and implementations into one module (for example `src/glmax/infer/solvers.py`).
- Consolidate covariance + hypothesis-test strategies into one module (for example `src/glmax/infer/inference.py`).
- Keep `src/glmax/infer/__init__.py` as compatibility export surface to avoid user-facing breakage.

**Dependencies:** Phase 1.

**Done when:** infer functionality is represented by cohesive modules with no residual one-function/pass-through module seams, while exports remain stable (covers `glmax-fit-pipeline-redesign.AC2.1`, `glmax-fit-pipeline-redesign.AC2.2`).
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: IRLS Fitter Objectization
**Goal:** Express IRLS as fitter-object behavior and remove optimize utility as a structural seam.

**Components:**
- Introduce/standardize fitter-object boundary for IRLS in `src/glmax/infer/fitters.py`.
- Move IRLS loop implementation under fitter ownership and keep solver/covariance/hypothesis composition explicit.
- Preserve current convergence and dispersion semantics through stable state contract.

**Dependencies:** Phase 2.

**Done when:** default fitting path executes through fitter object contract, and custom fitter strategy tests remain valid (covers `glmax-fit-pipeline-redesign.AC3.1`, `glmax-fit-pipeline-redesign.AC3.2`, `glmax-fit-pipeline-redesign.AC3.3`).
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: JAX Boundary Cleanup
**Goal:** Apply targeted JAX cleanup to boundary normalization without changing model semantics.

**Components:**
- Replace NumPy dtype probing at API boundaries with JAX-native subdtype checks (`jnp.issubdtype`) after boundary conversion.
- Standardize numeric boundary normalization and finite checks across canonical and compatibility entrypoints.
- Remove duplicated dtype-shape normalization branches that exist only for wrapper compatibility mechanics.

**Dependencies:** Phase 1 and Phase 3.

**Done when:** boundary dtype/finiteness behavior is preserved with JAX-first checks and dedicated regression coverage (covers `glmax-fit-pipeline-redesign.AC4.1`, `glmax-fit-pipeline-redesign.AC4.2`, `glmax-fit-pipeline-redesign.AC4.3`).
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Public Docstring and Contract Hardening
**Goal:** Bring public numerics entrypoint documentation to project standards and lock stable failure semantics.

**Components:**
- Update raw docstrings on public entrypoints (`glmax.fit`, `GLM.fit`, and public fitter/numerics surfaces) with required section labels.
- Document determinism/failure mode expectations where applicable.
- Align README/API migration notes with canonical workflow contract.

**Dependencies:** Phases 1-4.

**Done when:** public docstrings follow required section format and API documentation reflects canonical fit workflow and compatibility behavior (covers `glmax-fit-pipeline-redesign.AC5.1`, `glmax-fit-pipeline-redesign.AC5.2`).
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: Regression Verification and Compatibility Exit Criteria
**Goal:** Confirm behavioral parity and define readiness for later removal of temporary compatibility shims.

**Components:**
- Execute full fit/inference regression suite and verify no statistical behavior regressions.
- Record compatibility guarantees and deprecation checkpoints for wrapper behavior.
- Finalize risk log and implementation handoff notes.

**Dependencies:** Phases 1-5.

**Done when:** all verification commands pass and compatibility/deprecation notes are explicit and test-backed (covers `glmax-fit-pipeline-redesign.AC1.2`, `glmax-fit-pipeline-redesign.AC3.2`, `glmax-fit-pipeline-redesign.AC5.2`).
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
| Not in scope for this redesign | n/a | n/a | n/a |

### Planned Validation Experiments
| Experiment ID | Type (recovery/SBC/PPC) | Success Criterion | Notes |
| --- | --- | --- | --- |
| SIM-1 | n/a | n/a | Not in scope |

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | Consolidation may accidentally break import paths for downstream users who import internal infer modules directly | medium | Maintain re-exports in `infer/__init__.py` and add compatibility tests | implementation |
| R2 | IRLS migration into fitter object could alter update ordering if not copied exactly | high | Add regression tests that compare convergence metadata and coefficients before/after refactor | implementation |
| R3 | Aggressive cleanup could over-scope into algorithmic changes | medium | Keep DoD constrained to structural/API cleanup and parity validation | implementation |
| R4 | Ambiguity about timeline for removing compatibility shim in `GLM.fit` | low | Define explicit deprecation milestone after post-refactor release cycle | maintainers |

## Additional Considerations
- `python-module-design` guidance is intentionally applied: this plan consolidates micro-modules into a few stable responsibilities and avoids introducing new taxonomy-only files.
- If implementation discovers a better two-module split (instead of three or four) with equal clarity, consolidation is preferred over preserving planned filenames.

## Acceptance Criteria
### Canonical Workflow
- `glmax-fit-pipeline-redesign.AC1.1`: `glmax.fit` is the sole implementation owner of fit orchestration; `GLM.fit` is compatibility mapping only.
- `glmax-fit-pipeline-redesign.AC1.2`: For Gaussian, Poisson, Binomial, and Negative Binomial families, wrapper/direct output parity remains within existing tolerance baselines.

### Infer Consolidation
- `glmax-fit-pipeline-redesign.AC2.1`: Over-fragmented infer seams are consolidated into cohesive modules aligned to stable responsibilities.
- `glmax-fit-pipeline-redesign.AC2.2`: Public infer exports remain stable for supported imports; no new trivial wrapper modules are introduced.

### IRLS Fitter Contract
- `glmax-fit-pipeline-redesign.AC3.1`: IRLS executes through a fitter object boundary rather than standalone optimize utility plumbing.
- `glmax-fit-pipeline-redesign.AC3.2`: Existing fitter strategy injection behavior remains supported.
- `glmax-fit-pipeline-redesign.AC3.3`: Invalid strategy types still fail early with clear built-in exception messages.

### JAX Cleanup
- `glmax-fit-pipeline-redesign.AC4.1`: Boundary dtype validation uses JAX-native subdtype checks in canonical fit entrypoint paths.
- `glmax-fit-pipeline-redesign.AC4.2`: Non-numeric dtype inputs for `X`, `y`, or `offset` still raise deterministic boundary errors.
- `glmax-fit-pipeline-redesign.AC4.3`: Wrapper and direct entrypoints share one boundary normalization contract for dtype/shape/finiteness checks.

### Documentation and Contract Clarity
- `glmax-fit-pipeline-redesign.AC5.1`: Public numerics entrypoints use raw docstrings with required section labels.
- `glmax-fit-pipeline-redesign.AC5.2`: Failure-mode and compatibility semantics are documented and test-referenced in code/docs.

## Glossary
- Canonical entrypoint: The single public API surface that owns real workflow implementation (`glmax.fit`).
- Compatibility shim: A wrapper maintained for migration support that translates arguments but does not own core behavior.
- Fitter object: Strategy object that owns iterative fit execution contract (for example IRLS) and composes lower-level solver/inference strategies.
- Boundary validation: Input normalization and rejection logic performed before numerics execution.
- Cohesive module boundary: A module that owns one stable responsibility and avoids taxonomy-only fragmentation.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-03-04 | N/A | Draft | Plan created | |
| 2026-03-04 | Draft | Draft | Filled DoD, architecture, phases, and acceptance criteria | Codex |
