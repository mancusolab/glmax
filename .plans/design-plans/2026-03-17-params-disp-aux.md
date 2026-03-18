# Clarify GLM Dispersion Terminology and Params Aux Contract Design

## Status
Approved for Implementation

## Handoff Decision
- Current decision: approved
- Ready for implementation: yes
- Blocking items:
  - None.

## Metadata
- Date: 2026-03-17
- Slug: params-disp-aux
- Artifact Directory: `.plans/design-plans/artifacts/2026-03-17-params-disp-aux`

## Summary
This design introduces a breaking parameter-contract cleanup centered on `Params(beta, disp, aux)`. Its purpose is to stop overloading `Params.disp` across two different concepts: GLM or exponential-dispersion-model dispersion `phi`, and family-specific extra parameters such as Negative Binomial `alpha`. The design makes each field mean one thing consistently so the public API, family contracts, fitting, prediction, sampling, and inference all use the correct quantity.

The high-level approach keeps the current module boundaries and IRLS-based fitting stack, but changes the semantics and plumbing around parameters. Under the new contract, `disp` always means GLM dispersion, `aux` holds optional family-specific scalar state, fixed-dispersion families canonicalize `disp` to `1.0`, and Negative Binomial moves `alpha` into `aux`. Downstream inference then reads `fitted.params.disp` as the covariance-scaling quantity, while docs and tests are updated to remove the old overloaded meaning.

## Problem Statement
`glmax` currently overloads `Params.disp` across two different statistical concepts: the GLM/exponential-dispersion-model dispersion `phi`, and family-specific extra parameters such as Negative Binomial `alpha`. This leaks into public naming, family contracts, sampling and likelihood signatures, and downstream inference code. In particular, `GLM.scale(X, y, mu)` is already used as the covariance-scaling quantity in inference, while Negative Binomial stores `alpha` in `Params.disp` and still reports `scale() == 1.0`, making the public and internal semantics inconsistent.

The result is an interface that is harder to explain to users, harder to extend to additional families, and not aligned with how established GLM packages describe dispersion versus family-specific ancillary parameters. The design must simplify the user-facing parameter contract, separate GLM-wide dispersion from family-specific auxiliary terms, and ensure fitting, prediction, sampling, and inference all consume the correct quantity under the new semantics.

## Definition of Done
- A design defines a breaking parameter contract centered on `Params(beta, disp, aux)` and assigns one unambiguous meaning to each field.
- The design specifies that `disp` denotes the GLM/exponential-dispersion-model dispersion concept `phi`, while `aux` carries family-specific extra parameters when required.
- The design specifies that for `NegativeBinomial`, `disp` remains the canonical GLM value `1.0` and `aux` carries the fitted `alpha`.
- The design covers the required changes across model, family, fit, predict, infer, documentation, and tests so downstream inference remains statistically correct under the new contract.
- The design limits mandatory behavior to currently supported families and treats Tweedie-style extensibility as future-compatible context rather than current scope.

## Goals and Non-Goals
### Goals
- Make the public statistical vocabulary simple, intuitive, and formally correct for GLM users.
- Align `glmax` terminology with conventional GLM package usage closely enough to ease transition for power users.
- Separate GLM-wide dispersion semantics from family-specific auxiliary parameters without making the common case cumbersome.
- Preserve correct downstream inference behavior by making covariance-scaling and family-parameter semantics explicit.

### Non-Goals
- Add new families such as Tweedie in this change.
- Preserve backward compatibility or add deprecation warnings for the old `Params.disp` semantics.
- Redesign the broader grammar-first workflow outside the terminology and parameter-contract surface needed for this issue.

## Existing Patterns
Current `glmax` keeps the public grammar surface at package root and delegates implementation to stable internal seams:

- `src/glmax/_fit/types.py` owns `Params`, `FitResult`, `FittedGLM`, and `AbstractFitter`.
- `src/glmax/_fit/fit.py` owns the public `fit` and `predict` verbs.
- `src/glmax/_fit/irls.py` owns the default IRLS fitter and currently writes `Params(beta, disp)`.
- `src/glmax/family/dist.py` owns family likelihood, variance, dispersion handling, and Negative Binomial `alpha` updates.
- `src/glmax/_infer/stderr.py` and `src/glmax/_infer/hyptest.py` currently reconstruct inference scaling through `model.scale(X, y, mu)`.

This design keeps those module boundaries. It does not introduce a new parameter-contract file or a separate “semantics” package. The main intentional divergence is semantic, not structural: the current code treats `Params.disp` as both EDM dispersion and family-specific NB `alpha`. The new design splits those roles while keeping the existing fit/family/infer module layout.

## Model Acquisition Path
- Path: `existing-codebase-port`
- Why this path: the work redesigns terminology and statistical contracts inside an existing scientific codebase whose current behavior has already been investigated locally.
- User selection confirmation: confirmed in conversation while scoping the design around the current `glmax` parameter and inference contracts.

## Required Workflow States
- model_path_decided: yes
- codebase_investigation_complete_if_port: yes
- simulation_contract_complete_if_in_scope: n/a

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| SRC-1 | `src/glmax/_fit/types.py` | local-code | Current `Params` carrier and fit-result validation. | high |
| SRC-2 | `src/glmax/family/dist.py` | local-code | Current family contracts, including NB `alpha` stored as `disp`. | high |
| SRC-3 | `src/glmax/_infer/stderr.py` and `src/glmax/_infer/hyptest.py` | local-code | Current inference scaling behavior. | high |
| SRC-4 | https://stat.ethz.ch/R-manual/R-patched/library/stats/html/family.html | official-doc | Reference terminology for GLM family dispersion. | high |
| SRC-5 | https://stat.ethz.ch/R-manual/R-patched/RHOME/library/stats/html/summary.glm.html | official-doc | Reference treatment of GLM dispersion in summaries and inference. | high |
| SRC-6 | https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html | official-doc | Reference NB `alpha` as ancillary parameter separate from GLM scale. | high |

## Model Option Analysis (Required When `suggested-model`)
Not applicable. This plan refines terminology and contracts for the existing supported families rather than selecting a new model family.

## Existing Codebase Port Contract (Required When `existing-codebase-port`)
- Porting objective: preserve the current grammar-first workflow while replacing the overloaded `disp` semantics with a contract that separates EDM dispersion from family-specific auxiliary parameters.
- Source selection confirmation: use the current `glmax` worktree as the source of truth for module layout, tests, and supported families.

### Source Pin
| Source ID | Source Type (`local-directory` or `github-url`) | Path/URL | Commit/Tag | Notes |
| --- | --- | --- | --- | --- |
| PORT-SRC-1 | local-directory | /Users/nicholas/Projects/glmax | worktree-state-2026-03-17 | Unreleased package; breaking changes are allowed. |

### Behavior Inventory And Parity Targets
| Behavior ID | Surface (`cli`/`api`/`numerics`/`io`) | Current Behavior | Target Behavior | Evidence Plan (tests/golden outputs) |
| --- | --- | --- | --- | --- |
| PORT-BHV-1 | api | `Params` stores `(beta, disp)` and overloads `disp` across families. | `Params` stores `(beta, disp, aux)` with one stable meaning per field. | Contract tests in `tests/package/`, `tests/fit/`, and `tests/glm/`. |
| PORT-BHV-2 | numerics | Negative Binomial stores `alpha` in `disp`; inference separately uses `model.scale(...)`. | Negative Binomial stores `alpha` in `aux`; inference uses canonical fitted `params.disp`. | Family tests, inference tests, and reference-package checks. |
| PORT-BHV-3 | api | Family and GLM method signatures expose only `disp` for likelihood and sampling helpers. | Family and GLM contracts explicitly separate `disp` from `aux`. | `tests/glm/`, `tests/family/`, and docstring checks. |
| PORT-BHV-4 | api | Docs and tests mix public grammar usage with `_fit` internals for fitter-specific examples. | Public terminology and examples consistently describe `disp` vs `aux`; fitter internals stay advanced-only. | README/docs review plus package contract tests. |

## Codebase Investigation Findings (Required When `existing-codebase-port`)
- Investigation mode: `local-directory`
- Investigation completion: yes
- Investigator: `scientific-codebase-investigation-pass`

| Finding ID | Source Scope | Summary | Evidence (file:line or commit:path:line) | Status (`confirmed`/`discrepancy`/`addition`/`missing`) |
| --- | --- | --- | --- | --- |
| PORT-INV-1 | `src/glmax/_fit/types.py` | `Params` currently exposes only `beta` and `disp`; validation treats `disp` as a single scalar for all families. | `src/glmax/_fit/types.py:15` | confirmed |
| PORT-INV-2 | `src/glmax/family/dist.py` | Negative Binomial documents `disp` as overdispersion `alpha`, while `scale()` still returns `1.0`. | `src/glmax/family/dist.py:479`, `src/glmax/family/dist.py:500` | confirmed |
| PORT-INV-3 | `src/glmax/_infer/` | Inference currently derives `phi` through `model.scale(X, y, mu)` instead of reading a canonical fitted value from `Params`. | `src/glmax/_infer/stderr.py:48`, `src/glmax/_infer/hyptest.py:119` | confirmed |
| PORT-INV-4 | `src/glmax/glm.py` | `GLM` exposes both fit-time dispersion hooks and a public `scale(X, y, mu)` helper, which makes the source of truth for dispersion ambiguous. | `src/glmax/glm.py:147`, `src/glmax/glm.py:186` | discrepancy |
| PORT-INV-5 | docs/tests | Contract tests and docs assume the grammar-first top-level API, but some advanced docs/tests still import `_fit` internals for fitter-specific use. | `tests/package/test_api.py:82`, `docs/api/fitters.md:3` | addition |

## External Research Findings (When Triggered)
| Claim ID | Claim | Source URL | Source Type | Access Date | Confidence (high/med/low) |
| --- | --- | --- | --- | --- | --- |
| EXT-1 | Traditional GLM references treat dispersion/scale as the EDM-wide `phi`, with fixed `phi = 1` for Poisson and Binomial. | https://stat.ethz.ch/R-manual/R-patched/library/stats/html/family.html | official-doc | 2026-03-17 | high |
| EXT-2 | R `summary.glm` reports and estimates GLM dispersion separately from family-specific parameterizations. | https://stat.ethz.ch/R-manual/R-patched/RHOME/library/stats/html/summary.glm.html | official-doc | 2026-03-17 | high |
| EXT-3 | Statsmodels documents Negative Binomial `alpha` as an ancillary parameter rather than the generic GLM scale. | https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html | official-doc | 2026-03-17 | high |

## Mathematical Sanity Checks
- Summary: the refactor must preserve the distinction between EDM dispersion `phi` and family-specific auxiliary parameters. For Gaussian and Gamma, `Var(Y | mu)` remains `phi V(mu)`. For Poisson and Binomial, `phi` stays canonical `1.0`. For Negative Binomial, the variance remains `mu + alpha mu^2`, with `alpha` stored in `aux` rather than `disp`.
- Blocking issues: none identified at design time. The mathematics are unchanged; the risk is semantic misrouting in implementation.
- Accepted risks: `aux` is intentionally generic and scalar-oriented. If a future family requires structured auxiliary state, that will require a follow-on contract review rather than being solved in this design.

Detailed artifacts:
- `.plans/design-plans/artifacts/2026-03-17-params-disp-aux/model-symbol-table.md`
- `.plans/design-plans/artifacts/2026-03-17-params-disp-aux/equation-to-code-map.md`

## Solver Strategy Decision
- User preference: keep the current fitter and solver strategy unless the terminology cleanup requires otherwise.
- Chosen strategy: retain the current IRLS + `lineax` solver stack.
- Why this strategy: the problem is semantic, not solver-related. Changing solvers would add noise and make regression review harder.

## Solver Translation Feasibility
- Summary: no solver translation is required. The contract change should preserve the existing IRLS update structure and only split parameter flow into `disp` versus `aux`.
- Blocking constraints: the IRLS loop and family hooks currently assume a single scalar parameter in several call sites, so plumbing changes must be coordinated across fit and family methods.
- Custom-solver rationale (if chosen): not applicable.

Detailed artifact:
- `.plans/design-plans/artifacts/2026-03-17-params-disp-aux/solver-feasibility-matrix.md`

## Layer Contracts
### Ingress
- Contract: `Params` becomes the canonical fitted and warm-start carrier with fields `beta`, `disp`, and `aux`. `beta` is a finite inexact rank-1 coefficient vector. `disp` is a finite inexact scalar for EDM dispersion `phi`. `aux` is either `None` or a finite inexact scalar interpreted by the active family.
- Rejection rules: reject non-inexact or non-scalar `disp`/`aux`; reject `aux` for families that do not use it; reject family-specific invalid `aux` values such as non-positive NB `alpha`; keep `weights` unsupported as in the current fit/predict contracts.

### Pipeline
- Contract: fit-time canonicalization is family-aware. The active `GLM`/family boundary decides whether `disp` is estimated, fixed, or ignored, and whether `aux` is required, optional, or forbidden. `FitResult` and `FittedGLM` carry only canonicalized `Params`.
- Validation-first checks: validate `Params` shape/type at carrier boundaries; validate family compatibility before entering IRLS updates; canonicalize fixed-dispersion families to `disp = 1.0`; canonicalize unused `aux` to `None`.

### Numerics
- Contract: family likelihood, variance, weight, sampling, and update hooks conceptually accept both `disp` and `aux`. `update_dispersion`/`estimate_dispersion` remain responsible for EDM `phi`. New auxiliary hooks handle family-specific parameters such as NB `alpha`. Downstream inference reads `fitted.params.disp` as the covariance-scaling quantity.
- Result/status semantics: `fit()` returns canonical `Params(beta, disp, aux)` for all supported families; inference statistics remain shape-aligned with `beta`; auxiliary parameters may affect likelihood and variance but do not redefine the meaning of `disp`.

### Egress
- Contract: public docs, docstrings, and examples describe `disp` as GLM dispersion and `aux` as family-specific. Advanced fitter/solver docs may still mention `_fit` internals, but the primary workflow remains top-level `glmax.specify/fit/predict/infer/check`.
- Output/exit-code mapping: not a CLI change. Success remains the existing Python noun contracts; semantic failures raise deterministic validation errors.

## Data Conversion and Copy Strategy
Not applicable. This design does not change tabular ingestion, array conversion, or copy strategy.

## Multi-Input Reconciliation Contract (Required When Multiple Tabular Sources Feed Numerics)
Not applicable. The design does not add or change multi-input reconciliation behavior.

## Validation Strategy
- Boundary checks: package-root exports, `Params` construction, family-aware warm-start validation, and public verb signatures.
- Shape/range/domain checks: `beta` remains rank-1; `disp` remains finite scalar; `aux` is `None` or finite scalar; NB `aux` must be strictly positive; unused `aux` is rejected rather than silently ignored.
- Multi-input alignment checks (key uniqueness, overlap expectations, deterministic row ordering): not applicable.
- Failure semantics: deterministic `TypeError`/`ValueError` at the nearest public or family boundary; no silent reinterpretation of `disp` as a family-specific parameter.

## Testing and Verification Strategy
- TDD scope: update contract tests first for the new `Params(beta, disp, aux)` shape and family semantics, then update fit/infer/family behavior to satisfy them.
- Regression strategy: preserve current supported-family fit behavior and public workflow while changing only the parameter semantics. Add explicit tests that NB uses `aux` for `alpha` and that inference reads `params.disp` as `phi`.
- Verification commands:
  - `pytest -p no:capture tests/package/test_api.py tests/package/test_grammar.py`
  - `pytest -p no:capture tests/fit tests/infer tests/glm tests/family`
  - `pytest -p no:capture tests/data`

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Parameter Carrier Contract
**Goal:** Introduce the new fitted-parameter carrier and family-aware validation rules without changing the overall workflow.

**Components:**
- `Params`, `FitResult`, `FittedGLM`, and warm-start validation in `src/glmax/_fit/types.py` — extend the carrier to `(beta, disp, aux)`, validate scalar semantics, and expose forwarding/accessor updates needed by the rest of the package.
- Model-level parameter canonicalization in `src/glmax/glm.py` — add the family-aware hooks needed to validate and canonicalize `disp` and `aux` before fit/predict/infer consume them.

**Dependencies:** None.

**Done when:** `params-disp-aux.AC1.*` pass with updated package contract tests and the carrier remains a valid pytree/warm-start input across the public fit surface.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Family Semantics Split
**Goal:** Separate EDM dispersion from family-specific auxiliary parameters inside the family and GLM computation seams.

**Components:**
- Exponential-dispersion family base hooks in `src/glmax/family/dist.py` — distinguish EDM-dispersion responsibilities from auxiliary-parameter responsibilities and document which families use which inputs.
- Supported-family implementations in `src/glmax/family/dist.py` — keep Gaussian and Gamma on `disp`, keep Poisson and Binomial fixed at canonical `disp = 1.0`, and move Negative Binomial `alpha` into `aux`.
- GLM helper surface in `src/glmax/glm.py` — route likelihood, variance, weight, sampling, and parameter-estimation helpers through the split `disp`/`aux` contract. If `scale(X, y, mu)` remains as an implementation helper, it is no longer the public source of truth for fitted dispersion.

**Dependencies:** Phase 1.

**Done when:** `params-disp-aux.AC2.*` pass and supported families expose one unambiguous meaning for `disp` and `aux` in both code and docstrings.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Fit, Predict, and Inference Plumbing
**Goal:** Propagate the new parameter semantics through fitting, prediction, and downstream inference without changing the top-level workflow.

**Components:**
- Fit/predict orchestration in `src/glmax/_fit/fit.py` and `src/glmax/_fit/irls.py` — carry canonical `aux`, initialize family-specific auxiliary parameters when absent, and return canonical `Params(beta, disp, aux)` for all supported families.
- Inference estimators in `src/glmax/_infer/stderr.py` and `src/glmax/_infer/hyptest.py` — read `fitted.params.disp` as the covariance-scaling quantity and stop treating family helpers such as `scale()` as a separate source of fitted dispersion.
- Package-root contract touch points in `src/glmax/__init__.py` and related docstrings — keep exports stable while updating the meaning of the parameter carrier.

**Dependencies:** Phases 1-2.

**Done when:** `params-disp-aux.AC3.*` pass, supported-family fits still converge, and inference remains finite and shape-correct under the new contract.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Documentation and Regression Alignment
**Goal:** Align public documentation, advanced fitter docs, and regression tests with the new terminology and semantics.

**Components:**
- User-facing docs in `README.md`, `docs/index.md`, `docs/api/nouns.md`, `docs/api/verbs.md`, `docs/api/fitters.md`, and `docs/api/inference.md` — define `disp` as GLM dispersion and `aux` as family-specific.
- Contract and regression suites under `tests/package/`, `tests/fit/`, `tests/infer/`, `tests/glm/`, `tests/family/`, and `tests/data/` — update existing assertions from the old overloaded `disp` semantics to the new `disp`/`aux` split.
- Project context and API narrative files where terminology is part of the contributor contract, including `AGENTS.md`.

**Dependencies:** Phases 1-3.

**Done when:** `params-disp-aux.AC4.*` pass, the repository-standard pytest commands succeed, and the documented terminology matches the implemented parameter semantics.
<!-- END_PHASE_4 -->

## Simulation And Inference-Consistency Validation
- In scope: no
- Simulate entrypoint/signature: not added in this design.
- Inputs: not applicable.
- Outputs: not applicable.
- Seed/RNG policy: not applicable.

### Assumption Alignment
| Inference Assumption | Simulation Rule | Mismatch Risk | Mitigation |
| --- | --- | --- | --- |
| No new inferential model is introduced | No new simulation API is required | Semantic regression could still alter downstream SE/test behavior | Cover with targeted inference regression tests and reference-package comparisons |

### Planned Validation Experiments
| Experiment ID | Type (recovery/SBC/PPC) | Success Criterion | Notes |
| --- | --- | --- | --- |
| SIM-1 | n/a | No standalone simulation experiment required for this contract refactor | Use regression and reference-package tests instead |

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | `aux` is intentionally generic. It could become ambiguous again if future families need structured or multiple auxiliary parameters. | medium | Constrain `aux` to scalar-or-`None` in this design and revisit only if a future family truly needs more. | implementation |
| R2 | Removing or demoting `scale(X, y, mu)` as a public dispersion source may touch tests and internal assumptions beyond inference. | medium | Make the design explicit: inference reads `params.disp`; retain `scale()` only as an internal helper if needed. | implementation |
| R3 | Existing reference tests and docs may contain implicit assumptions that NB `params.disp` means `alpha`. | high | Update regression tests and docs in the same change; fail fast on stale semantics. | implementation |

## Additional Considerations
- **Future extensibility:** the `aux` slot is chosen partly because likely next-step families such as Tweedie also need one extra scalar parameter. This design does not add Tweedie, but it should avoid naming or validation choices that make a scalar `aux` impossible to reuse later.
- **No new module layer:** keep the current `_fit`, `_infer`, and `family` seams. The work is a contract cleanup, not a re-architecture into finer-grained packages.

## Acceptance Criteria
### `params-disp-aux.AC1`: `Params` carries one stable meaning per field
- **`params-disp-aux.AC1.1` Success:** `Params` stores `beta`, `disp`, and `aux`, and remains a valid pytree and warm-start carrier.
- **`params-disp-aux.AC1.2` Success:** `beta` remains the coefficient vector, `disp` remains the GLM/EDM dispersion scalar, and `aux` remains the optional family-specific scalar.
- **`params-disp-aux.AC1.3` Failure:** non-inexact or non-scalar `disp`/`aux` values raise deterministic validation errors.
- **`params-disp-aux.AC1.4` Failure:** non-`None` `aux` is rejected for families that do not use an auxiliary parameter.
- **`params-disp-aux.AC1.5` Success:** warm-start paths accept canonical `Params(beta, disp, aux)` and preserve values through `fit(...)` and `infer(...)`.

### `params-disp-aux.AC2`: family semantics split EDM dispersion from auxiliary parameters
- **`params-disp-aux.AC2.1` Success:** Gaussian and Gamma use `disp` as EDM dispersion and ignore `aux`.
- **`params-disp-aux.AC2.2` Success:** Poisson and Binomial canonicalize `disp` to `1.0` and require `aux is None`.
- **`params-disp-aux.AC2.3` Success:** Negative Binomial canonicalizes `disp` to `1.0` and uses `aux` as `alpha` in likelihood, variance, sampling, and fitting updates.
- **`params-disp-aux.AC2.4` Failure:** invalid NB `aux` values such as non-positive or non-finite `alpha` are rejected deterministically.
- **`params-disp-aux.AC2.5` Success:** family and `GLM` docstrings describe which of `disp` and `aux` each family uses, fixes, or ignores.

### `params-disp-aux.AC3`: fitting, prediction, and inference consume the correct parameter
- **`params-disp-aux.AC3.1` Success:** `fit(...)` returns canonical `Params(beta, disp, aux)` for every currently supported family.
- **`params-disp-aux.AC3.2` Success:** downstream inference uses `fitted.params.disp` as the covariance-scaling quantity `phi` and does not treat NB `aux` as GLM dispersion.
- **`params-disp-aux.AC3.3` Success:** `predict(...)` and GLM mean computations remain correct under the updated parameter carrier.
- **`params-disp-aux.AC3.4` Success:** Wald, Score, Fisher-information, and Huber-style inference outputs remain finite and shape-aligned under the new contract.
- **`params-disp-aux.AC3.5` Success:** supported-family regression checks continue to pass after the `disp`/`aux` split.

### `params-disp-aux.AC4`: public terminology and contributor guidance align with the new contract
- **`params-disp-aux.AC4.1` Success:** README, API docs, and package docstrings describe `disp` as GLM dispersion and `aux` as family-specific.
- **`params-disp-aux.AC4.2` Success:** Negative Binomial documentation describes `alpha` as the auxiliary parameter stored in `aux`, not in `disp`.
- **`params-disp-aux.AC4.3` Failure:** stale references that describe NB `params.disp` as `alpha` are removed from docs, tests, and contributor context.
- **`params-disp-aux.AC4.4` Success:** advanced fitter/solver docs may still mention `_fit` internals, but the primary user workflow remains the top-level grammar API.

## Glossary
- **`Params(beta, disp, aux)`**: The new canonical fitted-parameter and warm-start carrier, with one stable meaning for each field.
- **`beta`**: The finite inexact rank-1 coefficient vector in `Params`.
- **`disp`**: The finite inexact scalar representing GLM or EDM dispersion `phi`.
- **`aux`**: An optional family-specific finite scalar, or `None` when the active family does not use an auxiliary parameter.
- **Dispersion `phi`**: The GLM or exponential-dispersion-model covariance-scaling quantity that inference should consume from `fitted.params.disp`.
- **Auxiliary parameter**: A family-specific extra parameter that is not the GLM-wide dispersion, such as Negative Binomial `alpha`.
- **Exponential Dispersion Model (EDM)**: The GLM framing in which dispersion `phi` is distinct from the variance function and remains the shared meaning of `disp`.
- **Negative Binomial `alpha`**: The family-specific overdispersion parameter that this design moves out of `disp` and into `aux`.
- **Canonicalization**: Family-aware normalization of parameter values, such as fixing `disp = 1.0` for Poisson, Binomial, and Negative Binomial, and normalizing unused `aux` to `None`.
- **Warm start**: Reusing canonical `Params(beta, disp, aux)` as initialization for `fit(...)`.
- **Covariance-scaling quantity**: The role of fitted dispersion in downstream inference, which the design assigns to `params.disp`.
- **`scale(X, y, mu)`**: An existing helper whose status becomes secondary to the fitted parameter contract, since inference should no longer treat it as the public source of truth for dispersion.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-03-17 | N/A | Draft | Plan created | |
| 2026-03-18 | Draft | In Review | Plan review advanced through the status workflow. | Codex |
| 2026-03-18 | In Review | Approved for Implementation | Readiness validation passed and implementation was already complete. | Codex |
