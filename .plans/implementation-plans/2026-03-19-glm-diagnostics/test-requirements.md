# Test Requirements — GLM Diagnostics

## Skills Applied
- `scientific-house-style:writing-good-tests` -- loaded and applied
- `scientific-plan-execute:verification-before-completion` -- loaded and applied
- `scientific-house-style:jax-equinox-numerics` -- loaded and applied (numerics-transform and stability coverage)
- `scientific-house-style:property-based-testing` -- reviewed; not applied (no pure-function normalization surfaces requiring PBT; golden-value tests against statsmodels provide stronger coverage than generated inputs for this domain)
- `scientific-plan-execute:simulation-for-inference-validation` -- reviewed; not applicable (design plan marks simulation scope as `no`)

## Requirements Matrix

| Test ID | AC ID | Phase | Task IDs | Test Type | Test File | Test Function / Class | Command | Failure-First Evidence Required | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TEST-001 | glm-diagnostics.AC1.1 | 2 | 2.1 | unit | `tests/diagnostics/test_residuals.py` | `test_concrete_subclass_can_be_instantiated` | `pytest -p no:capture tests/diagnostics/test_residuals.py::test_concrete_subclass_can_be_instantiated` | yes | blocked |
| TEST-002 | glm-diagnostics.AC1.2 | 4 | 4.3 | integration | `tests/diagnostics/test_check.py` | `test_check_custom_diagnostics_single` | `pytest -p no:capture tests/diagnostics/test_check.py::test_check_custom_diagnostics_single` | yes | blocked |
| TEST-003 | glm-diagnostics.AC1.3 | 2 | 2.1 | unit | `tests/diagnostics/test_residuals.py` | `test_abstract_diagnostic_cannot_be_instantiated_directly` | `pytest -p no:capture tests/diagnostics/test_residuals.py::test_abstract_diagnostic_cannot_be_instantiated_directly` | yes | blocked |
| TEST-004 | glm-diagnostics.AC2.1 | 2 | 2.2 | numerics-regression | `tests/diagnostics/test_residuals.py` | `TestPearsonResidual::test_pearson_gaussian_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestPearsonResidual::test_pearson_gaussian_matches_statsmodels` | yes | blocked |
| TEST-005 | glm-diagnostics.AC2.1 | 2 | 2.2 | numerics-regression | `tests/diagnostics/test_residuals.py` | `TestPearsonResidual::test_pearson_poisson_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestPearsonResidual::test_pearson_poisson_matches_statsmodels` | yes | blocked |
| TEST-006 | glm-diagnostics.AC2.1 | 2 | 2.2 | numerics-regression | `tests/diagnostics/test_residuals.py` | `TestPearsonResidual::test_pearson_binomial_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestPearsonResidual::test_pearson_binomial_matches_statsmodels` | yes | blocked |
| TEST-007 | glm-diagnostics.AC2.1 | 2 | 2.2 | unit | `tests/diagnostics/test_residuals.py` | `TestPearsonResidual::test_pearson_shape_n` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestPearsonResidual::test_pearson_shape_n` | yes | blocked |
| TEST-008 | glm-diagnostics.AC2.2 | 2 | 2.2 | unit | `tests/diagnostics/test_residuals.py` | `TestPearsonResidual::test_pearson_all_finite` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestPearsonResidual::test_pearson_all_finite` | yes | blocked |
| TEST-009 | glm-diagnostics.AC3.1 | 2 | 2.3 | numerics-regression | `tests/diagnostics/test_residuals.py` | `TestDevianceResidual::test_deviance_gaussian_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestDevianceResidual::test_deviance_gaussian_matches_statsmodels` | yes | blocked |
| TEST-010 | glm-diagnostics.AC3.1 | 2 | 2.3 | numerics-regression | `tests/diagnostics/test_residuals.py` | `TestDevianceResidual::test_deviance_poisson_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestDevianceResidual::test_deviance_poisson_matches_statsmodels` | yes | blocked |
| TEST-011 | glm-diagnostics.AC3.1 | 2 | 2.3 | numerics-regression | `tests/diagnostics/test_residuals.py` | `TestDevianceResidual::test_deviance_binomial_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestDevianceResidual::test_deviance_binomial_matches_statsmodels` | yes | blocked |
| TEST-012 | glm-diagnostics.AC3.1 | 2 | 2.3 | unit | `tests/diagnostics/test_residuals.py` | `TestDevianceResidual::test_deviance_shape_n` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestDevianceResidual::test_deviance_shape_n` | yes | blocked |
| TEST-013 | glm-diagnostics.AC3.2 | 1, 2 | 1.2, 2.3 | unit | `tests/diagnostics/test_residuals.py` | `TestDevianceResidual::test_deviance_poisson_zero_y_finite` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestDevianceResidual::test_deviance_poisson_zero_y_finite` | yes | blocked |
| TEST-014 | glm-diagnostics.AC3.2 | 1 | 1.2 | unit | `tests/family/test_families.py` | `TestDevianceContribs::test_poisson_deviance_zero_y_is_finite` | `pytest -p no:capture tests/family/test_families.py::TestDevianceContribs::test_poisson_deviance_zero_y_is_finite` | yes | blocked |
| TEST-015 | glm-diagnostics.AC4.1 | 2 | 2.4 | unit | `tests/diagnostics/test_residuals.py` | `TestQuantileResidual::test_quantile_gaussian_all_finite` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestQuantileResidual::test_quantile_gaussian_all_finite` | yes | blocked |
| TEST-016 | glm-diagnostics.AC4.1 | 2 | 2.4 | unit | `tests/diagnostics/test_residuals.py` | `TestQuantileResidual::test_quantile_poisson_all_finite` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestQuantileResidual::test_quantile_poisson_all_finite` | yes | blocked |
| TEST-017 | glm-diagnostics.AC4.1 | 2 | 2.4 | unit | `tests/diagnostics/test_residuals.py` | `TestQuantileResidual::test_quantile_binomial_all_finite` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestQuantileResidual::test_quantile_binomial_all_finite` | yes | blocked |
| TEST-018 | glm-diagnostics.AC4.1 | 2 | 2.4 | unit | `tests/diagnostics/test_residuals.py` | `TestQuantileResidual::test_quantile_shape_n` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestQuantileResidual::test_quantile_shape_n` | yes | blocked |
| TEST-019 | glm-diagnostics.AC4.2 | 2 | 2.4 | numerics-regression | `tests/diagnostics/test_residuals.py` | `TestQuantileResidual::test_quantile_gaussian_equals_standardised_pearson` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestQuantileResidual::test_quantile_gaussian_equals_standardised_pearson` | yes | blocked |
| TEST-020 | glm-diagnostics.AC4.3 | 2 | 2.4 | unit | `tests/diagnostics/test_residuals.py` | `TestQuantileResidual::test_quantile_poisson_zero_y_finite` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestQuantileResidual::test_quantile_poisson_zero_y_finite` | yes | blocked |
| TEST-021 | glm-diagnostics.AC4.4 | 2 | 2.4 | unit | `tests/diagnostics/test_residuals.py` | `TestQuantileResidual::test_quantile_deterministic` | `pytest -p no:capture tests/diagnostics/test_residuals.py::TestQuantileResidual::test_quantile_deterministic` | yes | blocked |
| TEST-022 | glm-diagnostics.AC5.1 | 3 | 3.1 | numerics-regression | `tests/diagnostics/test_gof.py` | `TestGofStats::test_gof_deviance_gaussian_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_gof.py::TestGofStats::test_gof_deviance_gaussian_matches_statsmodels` | yes | blocked |
| TEST-023 | glm-diagnostics.AC5.1 | 3 | 3.1 | numerics-regression | `tests/diagnostics/test_gof.py` | `TestGofStats::test_gof_deviance_poisson_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_gof.py::TestGofStats::test_gof_deviance_poisson_matches_statsmodels` | yes | blocked |
| TEST-024 | glm-diagnostics.AC5.2 | 3 | 3.1 | numerics-regression | `tests/diagnostics/test_gof.py` | `TestGofStats::test_gof_aic_gaussian_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_gof.py::TestGofStats::test_gof_aic_gaussian_matches_statsmodels` | yes | blocked |
| TEST-025 | glm-diagnostics.AC5.2 | 3 | 3.1 | numerics-regression | `tests/diagnostics/test_gof.py` | `TestGofStats::test_gof_bic_gaussian_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_gof.py::TestGofStats::test_gof_bic_gaussian_matches_statsmodels` | yes | blocked |
| TEST-026 | glm-diagnostics.AC5.3 | 3 | 3.1 | unit | `tests/diagnostics/test_gof.py` | `TestGofStats::test_gof_df_resid_is_n_minus_p` | `pytest -p no:capture tests/diagnostics/test_gof.py::TestGofStats::test_gof_df_resid_is_n_minus_p` | yes | blocked |
| TEST-027 | glm-diagnostics.AC5.4 | 3 | 3.1 | unit | `tests/diagnostics/test_gof.py` | `TestGofStats::test_gof_pearson_chi2_equals_formula` | `pytest -p no:capture tests/diagnostics/test_gof.py::TestGofStats::test_gof_pearson_chi2_equals_formula` | yes | blocked |
| TEST-028 | glm-diagnostics.AC6.1 | 3 | 3.2 | numerics-regression | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_leverage_gaussian_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_influence.py::TestInfluenceStats::test_leverage_gaussian_matches_statsmodels` | yes | blocked |
| TEST-029 | glm-diagnostics.AC6.1 | 3 | 3.2 | numerics-regression | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_leverage_poisson_matches_statsmodels` | `pytest -p no:capture tests/diagnostics/test_influence.py::TestInfluenceStats::test_leverage_poisson_matches_statsmodels` | yes | blocked |
| TEST-030 | glm-diagnostics.AC6.2 | 3 | 3.2 | unit | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_leverage_in_open_unit_interval` | `pytest -p no:capture tests/diagnostics/test_influence.py::TestInfluenceStats::test_leverage_in_open_unit_interval` | yes | blocked |
| TEST-031 | glm-diagnostics.AC6.3 | 3 | 3.2 | unit | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_cooks_distance_nonnegative` | `pytest -p no:capture tests/diagnostics/test_influence.py::TestInfluenceStats::test_cooks_distance_nonnegative` | yes | blocked |
| TEST-032 | glm-diagnostics.AC6.4 | 3 | 3.2 | unit | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_cooks_distance_matches_formula` | `pytest -p no:capture tests/diagnostics/test_influence.py::TestInfluenceStats::test_cooks_distance_matches_formula` | yes | blocked |
| TEST-033 | glm-diagnostics.AC7.1 | 4 | 4.3 | integration | `tests/diagnostics/test_check.py` | `test_check_default_returns_5_tuple` | `pytest -p no:capture tests/diagnostics/test_check.py::test_check_default_returns_5_tuple` | yes | blocked |
| TEST-034 | glm-diagnostics.AC7.1 | 4 | 4.3 | integration | `tests/diagnostics/test_check.py` | `test_check_default_positional_types` | `pytest -p no:capture tests/diagnostics/test_check.py::test_check_default_positional_types` | yes | blocked |
| TEST-035 | glm-diagnostics.AC7.2 | 4 | 4.3 | integration | `tests/diagnostics/test_check.py` | `test_check_custom_diagnostics_single` | `pytest -p no:capture tests/diagnostics/test_check.py::test_check_custom_diagnostics_single` | yes | blocked |
| TEST-036 | glm-diagnostics.AC7.2 | 4 | 4.3 | integration | `tests/diagnostics/test_check.py` | `test_check_custom_diagnostics_two` | `pytest -p no:capture tests/diagnostics/test_check.py::test_check_custom_diagnostics_two` | yes | blocked |
| TEST-037 | glm-diagnostics.AC7.3 | 4 | 4.3 | integration | `tests/diagnostics/test_check.py` | `test_check_filter_jit_produces_same_result` | `pytest -p no:capture tests/diagnostics/test_check.py::test_check_filter_jit_produces_same_result` | yes | blocked |
| TEST-038 | glm-diagnostics.AC7.4 | 4 | 4.3 | unit | `tests/diagnostics/test_check.py` | `test_check_rejects_non_fitted_glm` | `pytest -p no:capture tests/diagnostics/test_check.py::test_check_rejects_non_fitted_glm` | yes | blocked |

### Prerequisite Tests (Phase 1 -- family methods)

These tests verify Phase 1 family API extensions that are prerequisites for Phase 2-3 diagnostic ACs. They do not map directly to a diagnostic AC but are required for correctness of `deviance_contribs` and `cdf` used by downstream diagnostics.

| Test ID | Prerequisite For | Phase | Task IDs | Test Type | Test File | Test Function / Class | Command | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TEST-P01 | AC2, AC3, AC5 | 1 | 1.2 | numerics-regression | `tests/family/test_families.py` | `TestCdf::test_gaussian_cdf_matches_scipy` | `pytest -p no:capture tests/family/test_families.py::TestCdf::test_gaussian_cdf_matches_scipy` | blocked |
| TEST-P02 | AC4 | 1 | 1.2 | numerics-regression | `tests/family/test_families.py` | `TestCdf::test_poisson_cdf_matches_scipy` | `pytest -p no:capture tests/family/test_families.py::TestCdf::test_poisson_cdf_matches_scipy` | blocked |
| TEST-P03 | AC3 | 1 | 1.2 | unit | `tests/family/test_families.py` | `TestDevianceContribs::test_gaussian_deviance_contribs_matches_formula` | `pytest -p no:capture tests/family/test_families.py::TestDevianceContribs::test_gaussian_deviance_contribs_matches_formula` | blocked |
| TEST-P04 | AC3 | 1 | 1.2 | unit | `tests/family/test_families.py` | `TestDevianceContribs::test_poisson_deviance_contribs_formula` | `pytest -p no:capture tests/family/test_families.py::TestDevianceContribs::test_poisson_deviance_contribs_formula` | blocked |
| TEST-P05 | AC4 | 1 | 1.3 | numerics-regression | `tests/family/test_families.py` | `TestCdfAllFamilies::test_binomial_cdf_matches_scipy` | `pytest -p no:capture tests/family/test_families.py::TestCdfAllFamilies::test_binomial_cdf_matches_scipy` | blocked |
| TEST-P06 | AC4 | 1 | 1.3 | numerics-regression | `tests/family/test_families.py` | `TestCdfAllFamilies::test_gamma_cdf_matches_scipy` | `pytest -p no:capture tests/family/test_families.py::TestCdfAllFamilies::test_gamma_cdf_matches_scipy` | blocked |
| TEST-P07 | AC4 | 1 | 1.3 | numerics-regression | `tests/family/test_families.py` | `TestCdfAllFamilies::test_nb_cdf_matches_scipy` | `pytest -p no:capture tests/family/test_families.py::TestCdfAllFamilies::test_nb_cdf_matches_scipy` | blocked |
| TEST-P08 | AC4 | 1 | 1.3 | unit | `tests/family/test_families.py` | `TestCdfAllFamilies::test_cdf_shape_n` (parametrized x5) | `pytest -p no:capture tests/family/test_families.py::TestCdfAllFamilies::test_cdf_shape_n` | blocked |
| TEST-P09 | AC3 | 1 | 1.3 | unit | `tests/family/test_families.py` | `TestCdfAllFamilies::test_deviance_contribs_shape_n` (parametrized x5) | `pytest -p no:capture tests/family/test_families.py::TestCdfAllFamilies::test_deviance_contribs_shape_n` | blocked |

### Supplementary Tests (strengthen coverage, not strictly AC-mapped)

| Test ID | Strengthens | Phase | Test Type | Test File | Test Function / Class | Rationale |
| --- | --- | --- | --- | --- | --- | --- |
| TEST-S01 | AC2.2, AC3.2 | 2 | unit | `tests/diagnostics/test_residuals.py` | `TestDevianceResidual::test_deviance_all_finite` | Finiteness across all three families |
| TEST-S02 | AC5 | 3 | unit | `tests/diagnostics/test_gof.py` | `TestGofStats::test_gof_stats_is_eqx_module` | Verifies return type contract |
| TEST-S03 | AC5 | 3 | unit | `tests/diagnostics/test_gof.py` | `TestGofStats::test_gof_all_fields_finite` | Finiteness of all GofStats fields |
| TEST-S04 | AC6 | 3 | unit | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_influence_returns_influence_stats` | Verifies return type contract |
| TEST-S05 | AC6 | 3 | unit | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_leverage_shape_n` | Shape contract |
| TEST-S06 | AC6 | 3 | unit | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_cooks_distance_shape_n` | Shape contract |
| TEST-S07 | AC6 | 3 | numerics-regression | `tests/diagnostics/test_influence.py` | `TestInfluenceStats::test_leverage_sums_to_p` | Mathematical invariant: trace(H) = p |
| TEST-S08 | AC7 | 4 | integration | `tests/diagnostics/test_check.py` | `test_check_default_all_outputs_finite` | End-to-end finiteness check |
| TEST-S09 | AC3.2, AC4.3 | 1 | unit | `tests/family/test_families.py` | `TestCdfAllFamilies::test_nb_deviance_zero_y_finite` | NB zero-y finiteness |
| TEST-S10 | AC3, AC5 | 1 | unit | `tests/family/test_families.py` | `TestDevianceContribs::test_deviance_contribs_nonnegative` (parametrized x2) | Non-negativity invariant |
| TEST-S11 | AC3 | 1 | unit | `tests/family/test_families.py` | `TestDevianceContribs::test_gaussian_deviance_sum_equals_rss` | Gaussian deviance = RSS identity |

## Test Types Used

- **unit**: single function/module behavior -- shape, type, boundary, finiteness checks
- **integration**: boundary crossing and API composition -- `check()` wiring, JIT compilation, custom diagnostic tuples
- **numerics-regression**: golden-value tests against statsmodels/scipy reference implementations -- Pearson residuals, deviance residuals, leverage, AIC/BIC, deviance, CDF values

## AC-to-Test Coverage Summary

| AC ID | AC Description | Test IDs | Coverage Status |
| --- | --- | --- | --- |
| glm-diagnostics.AC1.1 | AbstractDiagnostic is eqx.Module with Generic[T] and abstract diagnose | TEST-001 | covered |
| glm-diagnostics.AC1.2 | Concrete subclasses instantiate and pass to check() | TEST-002 | covered |
| glm-diagnostics.AC1.3 | Direct instantiation raises error | TEST-003 | covered |
| glm-diagnostics.AC2.1 | PearsonResidual matches statsmodels for Gaussian/Poisson/Binomial | TEST-004, TEST-005, TEST-006, TEST-007 | covered |
| glm-diagnostics.AC2.2 | PearsonResidual finite at mu boundaries | TEST-008 | covered |
| glm-diagnostics.AC3.1 | DevianceResidual matches statsmodels for Gaussian/Poisson/Binomial | TEST-009, TEST-010, TEST-011, TEST-012 | covered |
| glm-diagnostics.AC3.2 | Poisson y=0 finite deviance residual | TEST-013, TEST-014 | covered |
| glm-diagnostics.AC4.1 | QuantileResidual finite for all families | TEST-015, TEST-016, TEST-017, TEST-018 | covered |
| glm-diagnostics.AC4.2 | Gaussian quantile residual = standardised Pearson | TEST-019 | covered |
| glm-diagnostics.AC4.3 | CDF clamping prevents inf | TEST-020 | covered |
| glm-diagnostics.AC4.4 | Deterministic output (no PRNG) | TEST-021 | covered |
| glm-diagnostics.AC5.1 | GofStats.deviance matches statsmodels | TEST-022, TEST-023 | covered |
| glm-diagnostics.AC5.2 | GofStats.aic/bic match statsmodels | TEST-024, TEST-025 | covered |
| glm-diagnostics.AC5.3 | GofStats.df_resid = n - p | TEST-026 | covered |
| glm-diagnostics.AC5.4 | GofStats.pearson_chi2 = sum formula | TEST-027 | covered |
| glm-diagnostics.AC6.1 | Leverage matches statsmodels hat_matrix_diag | TEST-028, TEST-029 | covered |
| glm-diagnostics.AC6.2 | Leverage in (0, 1) | TEST-030 | covered |
| glm-diagnostics.AC6.3 | Cook's distance non-negative | TEST-031 | covered |
| glm-diagnostics.AC6.4 | Cook's distance matches formula | TEST-032 | covered |
| glm-diagnostics.AC7.1 | check() returns 5-tuple with default diagnostics | TEST-033, TEST-034 | covered |
| glm-diagnostics.AC7.2 | check() accepts custom diagnostics tuple | TEST-035, TEST-036 | covered |
| glm-diagnostics.AC7.3 | eqx.filter_jit(check) compiles and matches eager | TEST-037 | covered |
| glm-diagnostics.AC7.4 | check() raises TypeError for non-FittedGLM | TEST-038 | covered |

## Human Verification Requirements

The following criteria aspects require human judgment beyond what automated tests cover. These are documented here with justification and verification approach rather than mapped to automated tests.

| AC ID | Aspect | Justification | Human Verification Approach |
| --- | --- | --- | --- |
| glm-diagnostics.AC4.1 | Quantile residual finiteness across all five families | Automated tests cover Gaussian, Poisson, Binomial. Gamma and NegativeBinomial quantile residuals are tested for CDF correctness at the family level (TEST-P06, TEST-P07) but not end-to-end through QuantileResidual.diagnose. | Manually fit Gamma and NB models and call `QuantileResidual().diagnose(fitted)`, verify all outputs are finite. |
| glm-diagnostics.AC2.2 | Pearson residual at extreme mu boundaries | TEST-008 checks finiteness across standard fit data. Extreme boundary behavior (mu near 0 for Poisson, mu near 0 or 1 for Binomial) under JIT is difficult to test exhaustively with fixed golden values. | Construct FittedGLM with manually set mu values near family boundaries (e.g. mu=1e-10 for Poisson, mu=1e-10 and mu=1-1e-10 for Binomial). Verify no NaN under `eqx.filter_jit`. |
| glm-diagnostics.AC5.2 | AIC/BIC definition alignment with statsmodels | statsmodels has two BIC definitions (`bic` = deviance-based, `bic_llf` = log-likelihood-based). The implementation uses log-likelihood-based BIC. Phase 3 test (TEST-025) compares against `sm_result.bic_llf`. | Confirm that the BIC definition documented in GofStats docstring matches the formula tested, and that the statsmodels comparison target (`bic_llf`) is the correct one for `-2*ll + p*log(n)`. |
| All ACs | Public API surface exports | Automated tests import from `glmax.diagnostics` submodule. The `__init__.py` export updates (Phase 4, Task 2) need verification that all names are accessible from `glmax.*`. | Run `python -c "import glmax; print(glmax.PearsonResidual, glmax.DevianceResidual, glmax.QuantileResidual, glmax.GoodnessOfFit, glmax.GofStats, glmax.Influence, glmax.InfluenceStats, glmax.AbstractDiagnostic, glmax.DEFAULT_DIAGNOSTICS)"` and verify no ImportError. |

## Implementation Decision Rationalization

### Decision: Deterministic mid-quantile instead of randomised quantile residuals
- **Impact on testing:** No PRNG-related test complexity. AC4.4 (determinism) is trivially testable.
- **Trade-off:** Mid-quantile is an approximation for discrete families. This is documented in design plan risk R1 and accepted.
- **Test coverage:** TEST-021 explicitly verifies determinism by calling diagnose twice on the same FittedGLM.

### Decision: Recompute Cholesky in Influence rather than persisting from IRLS
- **Impact on testing:** No dependency on FitResult internals. Influence tests are self-contained.
- **Trade-off:** O(np^2) recomputation cost, accepted per design plan risk R2.
- **Test coverage:** TEST-028 and TEST-029 validate correctness against statsmodels. TEST-S07 validates the mathematical invariant trace(H) = p.

### Decision: statsmodels golden values as reference
- **Impact on testing:** Provides strong external validation. Requires statsmodels as a test dependency (already present transitively).
- **Test coverage:** 10 tests (TEST-004 through TEST-006, TEST-009 through TEST-011, TEST-022 through TEST-025, TEST-028, TEST-029) use statsmodels golden values.

### Decision: isinstance guard at check() boundary, not inside JIT trace
- **Impact on testing:** TypeError is raised at Python trace time. TEST-038 verifies this directly.
- **JAX compatibility:** isinstance is evaluated before JAX traces, so the guard works both with and without JIT. This is consistent with Equinox conventions.

### Decision: CDF clamping to [eps, 1-eps] before norm.ppf
- **Impact on testing:** Prevents inf output. TEST-020 exercises the y=0 Poisson case where F(y-1) = F(-1) = 0, triggering the clamp.
- **Trade-off:** Clamp introduces a small bias at extremes, accepted per design plan risk R3.

## Verification Commands

Full test suite command: `pytest -p no:capture tests/`

Phase-scoped commands:
- Phase 1: `pytest -p no:capture tests/family/`
- Phase 2: `pytest -p no:capture tests/diagnostics/test_residuals.py`
- Phase 3: `pytest -p no:capture tests/diagnostics/test_gof.py tests/diagnostics/test_influence.py`
- Phase 4: `pytest -p no:capture tests/diagnostics/test_check.py`
- All diagnostics: `pytest -p no:capture tests/diagnostics/`

## Hard Requirements Checklist

- [x] Every AC has at least one mapped test (verified: 22 ACs, all mapped)
- [ ] Every behavior-changing task has failure-first evidence (blocked: implementation not started)
- [ ] Phase completion is blocked when mapped tests are missing or failing (enforced at execution time)
- [x] No AC relies solely on undocumented manual checks
- [x] Human verification items are documented with justification and approach
