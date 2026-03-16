# Inferrer Abstraction for infer() Design

## Status
Draft

## Handoff Decision
- Current decision: blocked
- Ready for implementation: no
- Blocking items:
  - Pending plan review and explicit approval.

## Metadata
- Date: 2026-03-16
- Slug: inferrer-abstraction
- Artifact Directory: `docs/design-plans/artifacts/2026-03-16-inferrer-abstraction`

## Summary

`glmax` provides a pipeline of public verbs for fitting and interpreting generalised linear models: `specify → fit → predict → infer → check`. The `infer()` step currently produces hypothesis-test results — statistics and p-values — by hardcoding the Wald test. Standard-error estimation is already pluggable via an `AbstractStdErrEstimator` argument, but the test step that consumes those standard errors is not. This design introduces an `AbstractInferrer` extension point that mirrors the existing SE-estimator pattern: a caller can pass any `AbstractInferrer` to `infer()`, and the library ships two concrete implementations out of the box.

`WaldInferrer` migrates the current Wald logic and remains the default, preserving backward compatibility. `ScoreInferrer` is added as a second concrete implementation; it derives per-coefficient score (Rao) statistics directly from the fit artifacts — the GLM weights and score residuals at convergence — without needing standard errors, so it sets `InferenceResult.se` to NaN. The work also applies a field rename (`InferenceResult.z → stat`) to generalise the result type for any test statistic, expands the public export surface, and is delivered in three sequenced phases to keep each commit reviewable and independently green.

## Problem Statement
`infer()` hardcodes the Wald test for computing p-values. There is no way for callers to substitute a different hypothesis test (e.g. score/Rao test) without modifying library internals. The standard-error estimator is already pluggable via the `stderr` argument; the test step is not. This design adds an `AbstractInferrer` extension point so the test strategy is as swappable as the SE strategy.

## Definition of Done
- A new `AbstractInferrer` abstract `eqx.Module` base class with a `__call__(fitted, stderr) -> InferenceResult` contract; each inferrer decides whether to call `stderr`
- A concrete `WaldInferrer` default (current Wald logic migrated into it; calls `stderr` internally)
- A concrete `ScoreInferrer` as a second shipped example (ignores `stderr`; computes score statistic from fit artifacts)
- `infer()` signature updated to `infer(fitted, inferrer=DEFAULT_INFERRER, stderr=DEFAULT_STDERR)` — a one-liner that delegates to `inferrer(fitted, stderr)`
- `InferenceResult.z` renamed to `stat` — shape unchanged, meaning generalised to any test statistic
- `AbstractInferrer`, `WaldInferrer`, `ScoreInferrer`, `AbstractStdErrEstimator`, `FisherInfoError`, `HuberError` exported from the public surface
- Existing tests updated for renamed field; new tests cover both concrete inferrers

## Goals and Non-Goals
### Goals
- Make the hypothesis test in `infer()` pluggable, consistent with the existing `stderr` pluggability pattern
- Ship `WaldInferrer` (default) and `ScoreInferrer` as concrete implementations
- Keep the `InferenceResult` shape fixed across all inferrers

### Non-Goals
- Changing the SE estimation API (`AbstractStdErrEstimator`, `stderr` argument)
- Supporting tests that require a separate null-model fit (likelihood ratio test)
- Dispersion inference (tracked separately in next.txt)

## Existing Patterns
- `AbstractStdErrEstimator` (`infer/stderr.py`) — abstract `eqx.Module`, `__call__(fitted) -> Array` covariance; pattern to mirror for `AbstractInferrer`
- `Fitter` (`fit.py`) — `Protocol`-based strategy; `IRLSFitter` is the default; analogous role to `AbstractInferrer`
- `infer(fitted, stderr=DEFAULT_STDERR)` — current signature; `stderr` moves to second-keyword position after `inferrer`
- `InferenceResult` — `NamedTuple` with `(params, se, z, p)`; `z` to be renamed `stat`
- `FittedGLM` — input noun carrying `model` and `result`; passed into both `stderr` and `inferrer`
- All modules labelled Functional Core / Imperative Shell; `infer/` is Functional Core

## Model Acquisition Path
- Path: n/a (software API design, no statistical model acquisition)
- Why this path: This feature adds a pluggability layer to existing inference machinery; no new statistical model is introduced.
- User selection confirmation: n/a

## Required Workflow States
- model_path_decided: n/a
- codebase_investigation_complete_if_port: n/a
- simulation_contract_complete_if_in_scope: n/a

## Mathematical Sanity Checks
- Summary: `ScoreInferrer` implements the per-coefficient score (Rao) test for GLMs. The score vector is `U = X^T diag(glm_wt) score_residual / φ`, where `score_residual = (y − μ) g′(μ)` and `glm_wt = 1 / (V(μ) g′(μ)²)`. Fisher information diagonal is `I_jj = diag(X^T diag(glm_wt) X)_j / φ`. Per-coefficient statistic: `s_j = U_j / √I_jj = [X^T (glm_wt · score_residual)]_j / √(φ · [X^T diag(glm_wt) X]_jj)`. Under H₀: `s_j ~ N(0,1)`; two-sided p-values use the standard normal.
- Blocking issues: none
- Accepted risks: `ScoreInferrer` uses `glm_wt` from the last IRLS step and `φ` from `family.scale()`, consistent with how `FisherInfoError` normalises. For NB families where `φ` is estimated iteratively, the SE and score statistics share the same approximation risk already accepted elsewhere.

## Layer Contracts

### Ingress — `infer()`
- Contract: accepts `FittedGLM` (validated by `_matches_fitted_glm_shape`), `AbstractInferrer`, `AbstractStdErrEstimator`
- Rejection rules: raises `TypeError` if any argument fails its type check

### Pipeline — `AbstractInferrer.__call__`
- Contract: `(fitted: FittedGLM, stderr: AbstractStdErrEstimator) -> InferenceResult`; concrete inferrer is responsible for calling `stderr` if needed
- Validation-first: concrete inferrers must validate inputs they depend on (fitted structure, array finiteness) before computing

### Egress — `InferenceResult`
- Contract: `NamedTuple(params: Params, se: Array, stat: Array, p: Array)`; `se` is `NaN`-valued (shape `(p,)`) for inferrers that do not use SE; `stat` and `p` are always finite and valid

## Validation Strategy
- Boundary checks: `infer()` validates `isinstance(fitted, FittedGLM)`, `isinstance(inferrer, AbstractInferrer)`, `isinstance(stderr, AbstractStdErrEstimator)` before delegating
- Shape/range/domain checks: `p` values must be in `[0, 1]`; `stat` must be finite; `se` shape must be `(p,)` matching `params.beta`
- Failure semantics: raise `TypeError` for wrong-type inputs; `ValueError` for invalid array contents

## Testing and Verification Strategy
- TDD scope: write failing tests for `WaldInferrer` and `ScoreInferrer` before implementation; write failing tests for renamed `stat` field before rename commit
- Regression strategy: `WaldInferrer` output must match pre-refactor `infer()` numerically (same p-values to float64 precision)
- Verification commands: `pytest -p no:capture tests`

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Rename `InferenceResult.z` to `stat`
**Goal:** Apply the breaking rename in isolation before adding new abstractions, so no phase mixes the rename with new logic.

**Components:**
- `InferenceResult` in `src/glmax/infer/inference.py` — rename field `z` to `stat`
- `infer()` in `src/glmax/infer/inference.py` — update the assignment that writes to the renamed field
- `infer/__init__.py` — no change required (delegates transparently)
- All test files referencing `.z` — update to `.stat`

**Dependencies:** None (first phase)

**Done when:** All existing tests pass with no `.z` attribute references remaining in source or tests. `pytest -p no:capture tests` green.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Add `inferrer.py` with `AbstractInferrer`, `WaldInferrer`, `ScoreInferrer`
**Goal:** The new abstraction exists, is tested, and `WaldInferrer` produces results numerically identical to the pre-refactor `infer()` output.

**Components:**
- New `src/glmax/infer/inferrer.py` (Functional Core) — `AbstractInferrer` abstract `eqx.Module`; `WaldInferrer` migrates current `infer()` Wald logic; `ScoreInferrer` implements per-coefficient score test using `glm_wt`, `score_residual`, and `family.scale()` from fit artifacts; `DEFAULT_INFERRER = WaldInferrer()`
- `AbstractInferrer` contract: `__call__(fitted: FittedGLM, stderr: AbstractStdErrEstimator) -> InferenceResult`
- `WaldInferrer` calls `stderr(fitted)` internally; `ScoreInferrer` ignores `stderr` and sets `se = jnp.full(p, jnp.nan)`

**Dependencies:** Phase 1 (`InferenceResult.stat` must exist)

**Done when:** `WaldInferrer` p-values match pre-refactor `infer()` to float64 precision; `ScoreInferrer` returns finite `stat` and `p` in `[0,1]` with `se` all-NaN; type-contract tests confirm `AbstractInferrer` rejects non-`FittedGLM` inputs. `pytest -p no:capture tests` green.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Wire `infer()` and expand public surface
**Goal:** `infer()` accepts the `inferrer` argument, the shell is updated, and all new types are publicly accessible.

**Components:**
- `src/glmax/infer/inference.py` — update `infer()` to `infer(fitted, inferrer=DEFAULT_INFERRER, stderr=DEFAULT_STDERR)` delegating to `inferrer(fitted, stderr)`; guard: `isinstance(inferrer, AbstractInferrer)`
- `src/glmax/infer/__init__.py` — update shell to thread `inferrer` and `stderr` through; re-export `AbstractInferrer`, `WaldInferrer`, `ScoreInferrer`, `DEFAULT_INFERRER`
- `src/glmax/__init__.py` — add `AbstractInferrer`, `WaldInferrer`, `ScoreInferrer`, `AbstractStdErrEstimator`, `FisherInfoError`, `HuberError` to `__all__`

**Dependencies:** Phase 2 (`inferrer.py` and all concrete classes must exist)

**Done when:** `infer(fitted)` (no args) produces same result as before; `infer(fitted, inferrer=ScoreInferrer())` routes to score test; wrong-type `inferrer` raises `TypeError`; all six newly exported names importable from `glmax`. `pytest -p no:capture tests` green.
<!-- END_PHASE_3 -->

## Simulation And Inference-Consistency Validation
- In scope: no

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step | Owner |
| --- | --- | --- | --- | --- |
| R1 | `ScoreInferrer` uses `glm_wt` at convergence, not at the null. For tests of individual coefficients at MLE this is correct, but callers may expect a restricted-model score test. | low | Document that `ScoreInferrer` is an MLE-point score test, not a restricted-model Rao test. | |
| R2 | `InferenceResult.z → stat` is a breaking change for any downstream code using the attribute by name. | medium | Single-phase rename; note in changelog. No migration shim — YAGNI. | |
| R3 | `AbstractStdErrEstimator` and friends are now public; their `strict=True` eqx.Module constraint limits subclassing flexibility. | low | Accepted; follows existing `AbstractStdErrEstimator` precedent. | |

## Additional Considerations

**`se` field for `ScoreInferrer`:** `InferenceResult.se` is set to `jnp.full(p, jnp.nan)` by `ScoreInferrer` because no standard error is computed. Callers using `.se` downstream must handle NaN. This is intentional and documented on the class.

## Acceptance Criteria

### inferrer-abstraction.AC1: `InferenceResult.stat` field rename
- **inferrer-abstraction.AC1.1 Success:** `InferenceResult` instances expose `.stat`, not `.z`
- **inferrer-abstraction.AC1.2 Failure:** accessing `.z` on an `InferenceResult` raises `AttributeError`
- **inferrer-abstraction.AC1.3 Success:** `.stat` shape matches `.params.beta` shape `(p,)`

### inferrer-abstraction.AC2: `WaldInferrer` correctness
- **inferrer-abstraction.AC2.1 Success:** `WaldInferrer()(fitted, stderr)` returns `InferenceResult` with same p-values as pre-refactor `infer()` to float64 precision
- **inferrer-abstraction.AC2.2 Success:** Gaussian family uses t-distribution; all other families use standard normal
- **inferrer-abstraction.AC2.3 Success:** `WaldInferrer` calls `stderr(fitted)` internally and uses the resulting covariance to compute SE
- **inferrer-abstraction.AC2.4 Failure:** non-`FittedGLM` first arg raises `TypeError`

### inferrer-abstraction.AC3: `ScoreInferrer` correctness
- **inferrer-abstraction.AC3.1 Success:** `ScoreInferrer()(fitted, stderr)` returns `InferenceResult` with `stat` finite, `p` in `[0,1]`, `se` all-NaN
- **inferrer-abstraction.AC3.2 Success:** `ScoreInferrer` does not call `stderr` (verified via mock or counter)
- **inferrer-abstraction.AC3.3 Success:** `stat` shape matches `(p,)` for all supported families
- **inferrer-abstraction.AC3.4 Edge:** Gaussian family produces valid two-sided p-values from score statistic

### inferrer-abstraction.AC4: `infer()` signature and delegation
- **inferrer-abstraction.AC4.1 Success:** `infer(fitted)` with no extra args produces same result as pre-refactor
- **inferrer-abstraction.AC4.2 Success:** `infer(fitted, inferrer=ScoreInferrer())` routes to `ScoreInferrer`
- **inferrer-abstraction.AC4.3 Success:** `infer(fitted, stderr=HuberError())` passes `HuberError` into `WaldInferrer`
- **inferrer-abstraction.AC4.4 Failure:** `infer(fitted, inferrer=object())` raises `TypeError`
- **inferrer-abstraction.AC4.5 Failure:** `infer(fitted, stderr=object())` raises `TypeError`

### inferrer-abstraction.AC5: Public surface exports
- **inferrer-abstraction.AC5.1 Success:** `from glmax import AbstractInferrer, WaldInferrer, ScoreInferrer` succeeds
- **inferrer-abstraction.AC5.2 Success:** `from glmax import AbstractStdErrEstimator, FisherInfoError, HuberError` succeeds
- **inferrer-abstraction.AC5.3 Success:** all six names appear in `glmax.__all__`

## Glossary

- **GLM (Generalised Linear Model)**: A flexible extension of ordinary linear regression that allows the response variable to follow any distribution in the exponential family (e.g., Gaussian, Poisson, Binomial) via a link function connecting the linear predictor to the mean.
- **`infer()`**: The public verb in glmax that takes a fitted model and returns an `InferenceResult` containing coefficient estimates, standard errors, test statistics, and p-values.
- **`FittedGLM`**: The input noun passed to `infer()`; carries both the `GLM` model specification and the `FitResult` from the IRLS solver.
- **`InferenceResult`**: A `NamedTuple` holding `(params, se, stat, p)` — one row of inferential summary per coefficient.
- **Wald test**: A hypothesis test that measures how many standard errors a coefficient estimate is from zero: `stat = β̂_j / SE(β̂_j)`. Requires a standard-error estimate.
- **Score test (Rao test)**: A hypothesis test based on the gradient (score) of the log-likelihood evaluated at the MLE. Does not require a separate standard-error estimate; the statistic is normalised by the square root of the Fisher information diagonal.
- **Score residual / working residual**: `(y − μ) g′(μ)` — the residual in the working-response scale used by IRLS. Despite the name in the codebase, this is a working residual, not a true log-likelihood score residual.
- **GLM weight (`glm_wt`)**: `1 / (V(μ) g′(μ)²)` — the per-observation weight used in IRLS; appears in both the Fisher information and the score statistic.
- **Fisher information diagonal**: `diag(Xᵀ diag(glm_wt) X) / φ` — used to normalise the score vector into a standard-normal statistic.
- **Dispersion parameter (`φ`)**: A scalar scaling factor for the variance function; equals 1 for Poisson and Binomial, estimated from residuals for Gaussian and NegativeBinomial.
- **`AbstractInferrer`**: The new abstract base class introduced by this design; defines the `__call__(fitted, stderr) -> InferenceResult` contract.
- **`AbstractStdErrEstimator`**: The existing abstract base class for standard-error strategies; `FisherInfoError` (model-based) and `HuberError` (sandwich/robust) are the two concrete implementations.
- **`eqx.Module` (Equinox)**: A PyTree-aware base class from the Equinox library. Subclasses behave like frozen dataclasses compatible with JAX transformations. `strict=True` disallows undeclared attributes.
- **IRLS (Iteratively Reweighted Least Squares)**: The standard algorithm for fitting GLMs; each iteration solves a weighted least-squares problem using the current linearisation of the link function.
- **Functional Core / Imperative Shell**: An architectural pattern used throughout glmax. Pure, JAX-compatible functions live in the Functional Core; I/O, validation, and orchestration live in the Imperative Shell.
- **MLE (Maximum Likelihood Estimate)**: The parameter values that maximise the log-likelihood; the converged output of IRLS.
- **t-distribution vs. standard normal**: `WaldInferrer` uses a t-distribution for Gaussian families (estimated dispersion) and the standard normal for all other families (fixed dispersion).
- **YAGNI**: "You Aren't Gonna Need It" — justification for not adding a migration shim for the `z → stat` rename.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-03-16 | N/A | Draft | Plan created | |
