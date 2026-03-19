# GLM Diagnostics API Design

## Status
Draft

## Handoff Decision
- Current decision: blocked
- Ready for implementation: no
- Blocking items:
  - Pending plan review and explicit approval.

## Metadata
- Date: 2026-03-19
- Slug: glm-diagnostics
- Artifact Directory: `.plans/design-plans/artifacts/2026-03-19-glm-diagnostics`

## Summary
glmax implements GLMs as a grammar of composable JAX modules. While the library already supports fitting models via its `check` verb, that verb currently returns an empty placeholder with no computed values. This plan fills that gap by building a complete diagnostics subsystem: five concrete diagnostic classes — Pearson residuals, deviance residuals, quantile residuals, goodness-of-fit statistics, and influence measures — each implementing a shared `AbstractDiagnostic[T]` interface. Two prerequisite methods (`cdf` and `deviance_contribs`) are first added to each GLM family, then used by the diagnostic classes. The top-level `check()` function accepts any tuple of diagnostic instances and returns a positionally-matched tuple pytree, making it straightforward to apply transformations across all outputs with `jax.tree_util.tree_map`.

The design is shaped by two hard constraints: full JIT-compatibility and no modification to the existing `FitResult` contract. Every output is either a JAX array or an `eqx.Module` containing only JAX arrays, satisfying the pytree requirement. Where IRLS already factorised `XᵀWX` during fitting, `Influence` recomputes the Cholesky factor rather than persisting it, trading a bounded O(np²) cost for a clean API boundary. Quantile residuals use a deterministic mid-quantile approximation instead of the randomised variant, removing any PRNG dependency. Reference values from statsmodels are used throughout testing to verify correctness across Gaussian, Poisson, and Binomial families.

## Problem Statement
glmax exposes a `check` grammar verb that currently returns an empty `Diagnostics` placeholder. Users running GLMs in production JAX pipelines have no programmatic way to validate model fit, inspect residuals, or detect distributional misspecification. The diagnostics system must be JIT-compatible, extensible without modifying a central struct, and follow the same pluggable strategy pattern used for families and links.

## Definition of Done
1. `AbstractDiagnostic[T](eqx.Module, Generic[T])` base class with abstract method `diagnose(fitted: FittedGLM) -> T`.
2. Five concrete diagnostics: `PearsonResidual`, `DevianceResidual`, `QuantileResidual` (deterministic mid-quantile), `GoodnessOfFit` (returns `GofStats`), `Influence` (returns `InfluenceStats`).
3. `check(fitted, diagnostics=DEFAULT_DIAGNOSTICS)` decorated with `eqx.filter_jit`, returning a tuple pytree mapping 1:1 to the diagnostics passed.
4. All outputs are JAX arrays or `eqx.Module` subtypes — fully JIT-compatible.
5. Existing `check` stub and `Diagnostics` NamedTuple in `diagnostics.py` replaced by the new implementation.

## Goals and Non-Goals
### Goals
- Pluggable, extensible diagnostics under a single `AbstractDiagnostic[T]` interface
- JIT-compatible tuple pytree return — `check()` maps over the diagnostics tuple internally via `tuple(d.diagnose(fitted) for d in diagnostics)`
- Deterministic quantile residuals via mid-quantile approximation (no PRNG key required)
- Cover the three core diagnostic categories: residuals, goodness-of-fit, influence
- Mirror existing codebase patterns: `eqx.Module`, `strict=True`, `AbstractXxx` base classes

### Non-Goals
- Plotting or human-readable display layer
- Convergence diagnostics (already available directly on `FittedGLM.converged`, `num_iters`)
- Family-specific calibration tests (e.g. Hosmer–Lemeshow)
- Per-observation log-likelihood contributions as a standalone verb

## Existing Patterns
- All public nouns are `eqx.Module` with `strict=True`; base classes follow `AbstractXxx` naming
- Strategy pattern: `AbstractFamily`, `AbstractLink`, `AbstractFitter` are all `eqx.Module` subtypes passed as arguments
- `FittedGLM` carries `y`, `X`, `mu`, `eta`, `glm_wt`, `score_residual`, `params` (beta, disp, aux)
- `GLM.log_prob(y, eta, disp, aux)` provides per-observation log-likelihood — needed for deviance and AIC/BIC
- `GLM.working_weights(eta, disp, aux)` returns `(mu, g_deriv, weight)` — needed for Pearson residuals and leverage
- Cholesky factor from IRLS is not currently stored on `FitResult`; leverage computation will need `X`, `glm_wt`, and `params`
- All arrays are float64 (x64 mode enabled globally)

## Model Acquisition Path
- Path: n/a (computational library feature, not a statistical model)
- Why this path: diagnostics are deterministic calculations on existing fit artifacts

## Required Workflow States
- model_path_decided: n/a
- codebase_investigation_complete_if_port: n/a
- simulation_contract_complete_if_in_scope: n/a

## Model Specification Sources
| Source ID | Path/Link | Type | Notes | Confidence (high/med/low) |
| --- | --- | --- | --- | --- |
| SRC-1 | | | | |

## Model Option Analysis (Required When `suggested-model`)
| Candidate ID | Model Family | When It Fits | Key Assumptions | Failure Modes | Supporting Citation(s) | Selection Status |
| --- | --- | --- | --- | --- | --- | --- |
| MOD-1 | | | | | | selected/rejected |

## External Research Findings
| Claim ID | Claim | Source URL | Source Type | Access Date | Confidence |
| --- | --- | --- | --- | --- | --- |
| EXT-1 | Randomised quantile residuals (Dunn & Smyth 1996) are standard normal under the true model for any GLM family | https://www.jstor.org/stable/1390802 | paper | 2026-03-19 | high |
| EXT-2 | Mid-quantile approximation `Φ⁻¹((F(y) + F(y-1)) / 2)` is a valid deterministic alternative to randomised quantile residuals for discrete families | secondary | secondary | 2026-03-19 | med |
| EXT-3 | Leverage `h_ii = [W^{1/2} X (XᵀWX)⁻¹ Xᵀ W^{1/2}]_ii`; Cook's distance `D_i = r_i² h_i / (p (1-h_i)²)` in GLM context | McCullagh & Nelder (1989), §11 | paper | 2026-03-19 | high |

## Mathematical Sanity Checks
- Summary: Residual formulas are standard GLM theory. Leverage uses Cholesky of XᵀWX (already computed during IRLS but not persisted — recomputed in `Influence`). Mid-quantile formula has no known pathological edge cases for the supported families.
- Blocking issues: none
- Accepted risks: Mid-quantile quantile residuals for discrete families are an approximation (not exact uniform); adequate for production diagnostics but should be documented.

Detailed artifacts:
- `.plans/design-plans/artifacts/2026-03-19-glm-diagnostics/model-symbol-table.md`
- `.plans/design-plans/artifacts/2026-03-19-glm-diagnostics/equation-to-code-map.md`

## Solver Strategy Decision
- User preference: reuse existing Cholesky factorisation pattern from IRLS
- Chosen strategy: recompute `chol(XᵀWX)` inside `Influence.diagnose` using `jax.scipy.linalg.cholesky` + `solve_triangular`
- Why this strategy: Cholesky is not persisted on `FitResult`; recomputing is O(np²) and JIT-safe. Persisting it would require changing the `FitResult` contract, out of scope.

## Solver Translation Feasibility
- Summary: leverage via `Z = solve_triangular(L, (W^{1/2} X)ᵀ, lower=True).T; h = sum(Z², axis=1)` is numerically stable and JIT-compatible
- Blocking constraints: none
- Custom-solver rationale: n/a

## Layer Contracts

### Ingress
- Contract: `check(fitted: FittedGLM, diagnostics: tuple[AbstractDiagnostic, ...] = DEFAULT_DIAGNOSTICS) -> tuple`
- Rejection rules: `isinstance(fitted, FittedGLM)` guard raises `TypeError` if violated; diagnostics tuple must be non-empty (runtime assertion)

### Numerics
- Contract: each `AbstractDiagnostic.diagnose(fitted)` receives the full `FittedGLM` and returns either a JAX array or an `eqx.Module` containing only JAX arrays
- Result/status semantics: no status channel — diagnostics raise on invalid input, return arrays on success; NaN propagation follows JAX default behaviour

### Egress
- Contract: `tuple[T, ...]` where each element `T` corresponds positionally to the input `diagnostics` tuple; structure is static and JAX-pytree-traversable
- Output mapping: `check()` iterates over the `diagnostics` tuple, calling `diagnose(fitted)` on each; returns a positional tuple of results

## Validation Strategy
- Boundary checks: `isinstance(fitted, FittedGLM)` at `check()` entry; family CDF outputs clamped to `[ε, 1-ε]` before `jax.scipy.stats.norm.ppf` in `QuantileResidual`
- Shape/range/domain checks: all residual arrays must be shape `(n,)`; leverage values must be in `(0, 1)`; Cook's distance non-negative
- Failure semantics: `TypeError` for wrong input type; otherwise NaN propagates silently (consistent with JAX convention)

## Testing and Verification Strategy
- TDD scope: each concrete diagnostic tested against statsmodels reference values for at least Gaussian, Poisson, and Binomial families
- Regression strategy: golden-value tests using statsmodels GLM for Pearson residuals, deviance residuals, leverage, AIC, BIC, deviance
- Verification commands: `pytest -p no:capture tests/diagnostics/`

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Family API extensions
**Goal:** Add `cdf` and deviance-contribution methods to `AbstractFamily` and all five concrete families, providing the mathematical building blocks needed by residual diagnostics.

**Components:**
- `AbstractFamily` in `src/glmax/family/dist.py` — add abstract methods `cdf(y, mu, disp, aux) -> Array` (CDF evaluated at y) and `deviance_contribs(y, mu, disp, aux) -> Array` (per-observation `2*(log_lik_sat - log_lik_fit)`)
- All five family implementations (`Gaussian`, `Poisson`, `Binomial`, `Gamma`, `NegativeBinomial`) in the same file — implement both methods; `Gaussian` and `Gamma` CDFs are exact; discrete families use the standard CDFs from `jax.scipy.stats`

**Dependencies:** None (extends existing module)

**Done when:** All five families implement `cdf` and `deviance_contribs`; tests compare against `scipy.stats` reference values for each family; `pytest -p no:capture tests/` passes
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: AbstractDiagnostic and residual diagnostics
**Goal:** Establish the pluggable diagnostic interface and implement the three residual diagnostic classes.

**Components:**
- `AbstractDiagnostic[T]` base class in `src/glmax/diagnostics.py` — `eqx.Module`, `Generic[T]`, abstract `diagnose(fitted: FittedGLM) -> T`
- `PearsonResidual(AbstractDiagnostic[Array])` — `(y - mu) / sqrt(V(mu))` using `GLM.working_weights`
- `DevianceResidual(AbstractDiagnostic[Array])` — `sign(y - mu) * sqrt(deviance_contribs)` using Phase 1 family method
- `QuantileResidual(AbstractDiagnostic[Array])` — `Φ⁻¹((F(y) + F(y-1)) / 2)` using Phase 1 `cdf` method; clamp CDF output to `[ε, 1-ε]` before `norm.ppf`

**Dependencies:** Phase 1 (family `cdf` and `deviance_contribs`)

**Done when:** Three residual classes produce arrays of shape `(n,)` matching statsmodels reference values to reasonable tolerance; tests cover Gaussian, Poisson, Binomial families; `pytest -p no:capture tests/diagnostics/` passes
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Structured diagnostics — GoodnessOfFit and Influence
**Goal:** Implement the two structured diagnostics that return typed result containers rather than raw arrays.

**Components:**
- `GofStats(eqx.Module, strict=True)` in `src/glmax/diagnostics.py` — fields: `deviance`, `pearson_chi2`, `df_resid`, `dispersion`, `aic`, `bic` (all scalar arrays)
- `GoodnessOfFit(AbstractDiagnostic[GofStats])` — computes all `GofStats` fields from `FittedGLM`; AIC/BIC use `GLM.log_prob`
- `InfluenceStats(eqx.Module, strict=True)` in `src/glmax/diagnostics.py` — fields: `leverage` `(n,)`, `cooks_distance` `(n,)`
- `Influence(AbstractDiagnostic[InfluenceStats])` — recomputes `chol(XᵀWX)` to obtain leverage via `solve_triangular`; Cook's distance from leverage and Pearson residuals

**Dependencies:** Phase 2 (uses `PearsonResidual` internally for Cook's distance)

**Done when:** `GofStats` fields match statsmodels reference for Gaussian and Poisson; leverage matches `statsmodels.GLMResults.get_influence().hat_matrix_diag` to float64 tolerance; `pytest -p no:capture tests/diagnostics/` passes
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: check() wiring and public API
**Goal:** Wire `check()` with `eqx.filter_jit`, define `DEFAULT_DIAGNOSTICS`, remove the old stub, and export everything through the public API.

**Components:**
- `DEFAULT_DIAGNOSTICS` constant in `src/glmax/diagnostics.py` — `(PearsonResidual(), DevianceResidual(), QuantileResidual(), GoodnessOfFit(), Influence())`
- `check(fitted, diagnostics=DEFAULT_DIAGNOSTICS)` in `src/glmax/diagnostics.py` — decorated with `@eqx.filter_jit`; replaces the existing stub; returns `tuple(d.diagnose(fitted) for d in diagnostics)`
- `src/glmax/__init__.py` — export `AbstractDiagnostic`, `PearsonResidual`, `DevianceResidual`, `QuantileResidual`, `GoodnessOfFit`, `GofStats`, `Influence`, `InfluenceStats`

**Dependencies:** Phases 1–3

**Done when:** `glmax.check(fitted)` returns a 5-tuple; `eqx.filter_jit(glmax.check)(fitted)` compiles and produces correct outputs; custom diagnostics tuple accepted; `pytest -p no:capture tests/` passes
<!-- END_PHASE_4 -->

## Simulation And Inference-Consistency Validation
- In scope: no

## Risks and Open Questions
| ID | Risk or Question | Severity | Mitigation or Next Step |
| --- | --- | --- | --- |
| R1 | Mid-quantile residuals for discrete families deviate from uniform for extreme y values | low | Document the approximation; exact randomised residuals can be added later as `RandomisedQuantileResidual` |
| R2 | Recomputing `chol(XᵀWX)` in `Influence` is a second factorisation (IRLS already did one) | low | Accepted cost; persisting the factor would require changing `FitResult` contract |
| R3 | `QuantileResidual` for `y=0` in Poisson uses `F(-1) = 0`; `Φ⁻¹(ε)` may be a large negative number | low | Clamping to `[ε, 1-ε]` before `norm.ppf` prevents ±inf |

## Additional Considerations

**Extensibility:** Users can subclass `AbstractDiagnostic[T]` to add custom diagnostics (e.g. standardised residuals, DFBETAS) without modifying any existing class. The `check()` signature accepts any tuple of `AbstractDiagnostic` instances.

**Discrete y edge cases:** `DevianceResidual` for Poisson with `y=0` uses the convention `0 * log(0/mu) = 0` — this must be enforced via `jnp.where` to avoid NaN under JIT.

## Acceptance Criteria

### glm-diagnostics.AC1: AbstractDiagnostic base class
- **glm-diagnostics.AC1.1 Success:** `AbstractDiagnostic` is an `eqx.Module` subclass parameterised by `Generic[T]` with abstract method `diagnose(fitted: FittedGLM) -> T`
- **glm-diagnostics.AC1.2 Success:** Concrete subclasses that implement `diagnose` can be instantiated and passed to `check()`
- **glm-diagnostics.AC1.3 Failure:** Instantiating `AbstractDiagnostic` directly (without subclassing) raises an error

### glm-diagnostics.AC2: PearsonResidual
- **glm-diagnostics.AC2.1 Success:** Returns array of shape `(n,)` equal to `(y - mu) / sqrt(V(mu))` matching statsmodels `resid_pearson` for Gaussian, Poisson, and Binomial families
- **glm-diagnostics.AC2.2 Edge:** Produces correct values when `mu` is close to 0 or to the boundary of the family's support (no NaN under JIT)

### glm-diagnostics.AC3: DevianceResidual
- **glm-diagnostics.AC3.1 Success:** Returns array of shape `(n,)` equal to `sign(y - mu) * sqrt(deviance_contribution_i)` matching statsmodels `resid_deviance` for Gaussian, Poisson, and Binomial families
- **glm-diagnostics.AC3.2 Edge:** Poisson with `y=0` produces a finite (non-NaN) value using `0 * log(0/mu) = 0` convention

### glm-diagnostics.AC4: QuantileResidual
- **glm-diagnostics.AC4.1 Success:** Returns array of shape `(n,)` that is finite for all observations across all five families
- **glm-diagnostics.AC4.2 Success:** For Gaussian family, quantile residuals equal standardised Pearson residuals (exact, since CDF is the normal CDF)
- **glm-diagnostics.AC4.3 Edge:** CDF values at 0 or 1 boundaries are clamped before `norm.ppf`; output is finite (not ±inf)
- **glm-diagnostics.AC4.4 Success:** Deterministic — same `FittedGLM` always produces the same quantile residuals with no PRNG key required

### glm-diagnostics.AC5: GoodnessOfFit / GofStats
- **glm-diagnostics.AC5.1 Success:** `GofStats.deviance` matches statsmodels `deviance` for Gaussian and Poisson
- **glm-diagnostics.AC5.2 Success:** `GofStats.aic` matches statsmodels `aic`; `GofStats.bic` matches statsmodels `bic`
- **glm-diagnostics.AC5.3 Success:** `GofStats.df_resid` equals `n - p`
- **glm-diagnostics.AC5.4 Success:** `GofStats.pearson_chi2` equals `sum((y - mu)² / V(mu))`

### glm-diagnostics.AC6: Influence / InfluenceStats
- **glm-diagnostics.AC6.1 Success:** `InfluenceStats.leverage` matches statsmodels `get_influence().hat_matrix_diag` for Gaussian and Poisson to float64 tolerance
- **glm-diagnostics.AC6.2 Success:** All leverage values satisfy `0 < h_i < 1`
- **glm-diagnostics.AC6.3 Success:** `InfluenceStats.cooks_distance` is non-negative for all observations
- **glm-diagnostics.AC6.4 Success:** Cook's distance equals `pearson_r_i² * h_i / (p * (1 - h_i)²)`

### glm-diagnostics.AC7: check() function
- **glm-diagnostics.AC7.1 Success:** `glmax.check(fitted)` with default diagnostics returns a 5-tuple mapping positionally to `(PearsonResidual, DevianceResidual, QuantileResidual, GoodnessOfFit, Influence)` results
- **glm-diagnostics.AC7.2 Success:** `glmax.check(fitted, diagnostics=(PearsonResidual(),))` returns a 1-tuple containing only Pearson residuals
- **glm-diagnostics.AC7.3 Success:** `eqx.filter_jit(glmax.check)(fitted)` compiles and produces the same results as the non-JIT call
- **glm-diagnostics.AC7.4 Failure:** `glmax.check("not_a_fitted_glm")` raises `TypeError`

## Glossary
- **GLM (Generalised Linear Model)**: A regression framework that extends ordinary linear regression to response variables with non-normal distributions, via a link function and an exponential-family distribution.
- **IRLS (Iteratively Reweighted Least Squares)**: The standard numerical algorithm for fitting GLMs; iteratively solves a weighted least-squares problem until convergence.
- **FittedGLM**: The glmax struct produced after fitting; carries the observed response `y`, design matrix `X`, fitted mean `mu`, linear predictor `eta`, weights, residuals, and model parameters.
- **`eqx.Module` / Equinox**: A JAX-compatible scientific-computing library. `eqx.Module` is its base class for typed, immutable structs that participate in JAX's pytree system.
- **`eqx.filter_jit`**: Equinox's JIT wrapper that traces only JAX array leaves and treats non-array fields as static, avoiding tracing errors that arise with plain `jax.jit`.
- **pytree**: JAX's term for any nested container (tuple, list, dict, `eqx.Module`) whose array leaves JAX can traverse uniformly — required for JIT and automatic differentiation.
- **`strict=True`** (Equinox): Module option that disallows dynamic field assignment after construction, enforcing immutability.
- **AbstractDiagnostic[T]**: The pluggable strategy interface introduced by this plan; each concrete diagnostic encapsulates one computation and returns a typed result `T`.
- **Pearson residual**: Observation-level residual normalised by the square root of the variance function: `(y − μ) / sqrt(V(μ))`.
- **Deviance residual**: Signed square-root of each observation's contribution to the total deviance: `sign(y − μ) * sqrt(d_i)`.
- **Quantile residual**: Residual constructed by mapping the fitted CDF through the standard normal quantile function; standard normal under the true model (Dunn & Smyth 1996).
- **Mid-quantile approximation**: A deterministic variant of quantile residuals for discrete responses that averages `F(y)` and `F(y−1)` before applying `Φ⁻¹`, avoiding the need for random number generation.
- **Deviance**: A goodness-of-fit statistic equal to twice the log-likelihood ratio between the saturated model and the fitted model.
- **Leverage (`h_ii`)**: The diagonal of the hat matrix; measures how much influence observation `i` has on its own fitted value.
- **Cook's distance**: A scalar influence measure per observation combining leverage and Pearson residuals; large values flag potentially influential points.
- **GofStats**: The glmax result struct for `GoodnessOfFit`; contains deviance, Pearson chi-squared, residual degrees of freedom, dispersion, AIC, and BIC as scalar arrays.
- **InfluenceStats**: The glmax result struct for `Influence`; contains per-observation leverage and Cook's distance as `(n,)` arrays.
- **AIC / BIC**: Penalised log-likelihood statistics used to compare model fit while discouraging overfitting.
- **`norm.ppf` / `Φ⁻¹`**: The inverse CDF of the standard normal distribution; maps uniform-scale CDF values to a normal scale in quantile residuals.
- **`solve_triangular`**: A numerical routine that solves a triangular linear system efficiently; used to compute leverage from the Cholesky factor without forming the full inverse.
- **Strategy pattern**: A design pattern where a family of interchangeable algorithms share a common interface and are passed as arguments; used throughout glmax for families, links, fitters, and diagnostics.
- **FitResult**: The internal glmax struct representing raw solver output; distinct from `FittedGLM` which is the user-facing fit object.

## Status Transition Log
| Date | From | To | Why | By |
| --- | --- | --- | --- | --- |
| 2026-03-19 | N/A | Draft | Plan created | |
