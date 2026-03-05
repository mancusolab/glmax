# Test Plan: GLMAX Fit Pipeline Redesign

## Scope

- Implementation plan: `docs/implementation-plans/2026-03-04-glmax-fit-pipeline-redesign/`
- Design plan: `docs/design-plans/2026-03-04-glmax-fit-pipeline-redesign.md`
- Validation date: 2026-03-05

## Automated Coverage Validation

Coverage result: **PASS**

| Acceptance Criterion | Required Test File(s) | Coverage Evidence | Status |
| --- | --- | --- | --- |
| AC1.1 canonical orchestration owner | `tests/test_fit_api.py`, `tests/test_glm.py` | `test_package_root_fit_export_identity`, `test_glm_fit_delegates_to_module_entrypoint` | PASS |
| AC1.2 wrapper/direct parity | `tests/test_fit_api.py`, `tests/test_glm.py` | `test_wrapper_and_canonical_fit_parity`, `test_valid_input_parity_is_preserved_after_boundary_cleanup`, `test_wrapper_and_canonical_remain_aligned_with_numpy_inputs` | PASS |
| AC2.1 infer internals consolidation | `tests/test_glm.py`, `tests/test_fit_api.py` | `test_consolidated_infer_modules_are_available`, `test_canonical_fit_routes_pvalues_through_inference_strategy` | PASS |
| AC2.2 public infer export stability | `tests/test_glm.py`, `tests/test_fit_api.py` | `test_infer_public_exports_remain_stable`, `test_legacy_infer_modules_warn_but_preserve_aliases` | PASS |
| AC3.1 fitter object boundary | `tests/test_fitters.py`, `tests/test_glm.py`, `tests/test_fit_api.py` | `test_fitters_module_provides_irls_contract`, `test_custom_fitter_strategy_injection_for_canonical_and_glm` | PASS |
| AC3.2 fitter strategy injection | `tests/test_fitters.py`, `tests/test_fit_api.py` | `test_custom_fitter_strategy_injection_for_canonical_and_glm`, `test_equivalent_custom_fitter_preserves_regression_parity` | PASS |
| AC3.3 invalid fitter failure | `tests/test_fitters.py` | `test_canonical_fit_rejects_invalid_fitter_type`, `test_glm_fit_rejects_invalid_override_fitter_type` | PASS |
| AC4.1 JAX-native dtype validation | `tests/test_fit_api.py` | `test_boundary_rejects_non_numeric_dtypes_with_deterministic_errors` | PASS |
| AC4.2 deterministic boundary exceptions | `tests/test_fit_api.py`, `tests/test_glm.py` | `test_boundary_rejects_non_numeric_dtypes_with_deterministic_errors`, `test_boundary_rejects_invalid_rank_shape_and_finiteness`, `test_wrapper_and_canonical_match_offset_finiteness_boundary_error` | PASS |
| AC4.3 shared boundary contract | `tests/test_fit_api.py`, `tests/test_glm.py` | `test_wrapper_and_canonical_share_boundary_normalization`, `test_wrapper_and_canonical_match_offset_finiteness_boundary_error` | PASS |
| AC5.1 required raw docstring sections | `tests/test_fit_api.py`, `tests/test_fitters.py`, `tests/test_glm.py` | `test_public_entrypoint_docstrings_follow_contract_sections`, `test_fitter_and_inference_docstrings_follow_contract_sections`, `test_contract_and_solver_docstrings_follow_required_sections` | PASS |
| AC5.2 docs/failure semantics linked to tests | `tests/test_fit_api.py`, `tests/test_fitters.py`, `tests/test_glm.py` | `test_docs_claim_canonical_workflow_is_backed_by_parity_assertions`, `test_docs_claim_invalid_fitter_type_is_enforced`, `test_docs_claim_shared_boundary_failures_is_backed_by_glm_checks` | PASS |

## Executed Verification Commands

All required commands from `test-requirements.md` passed:

1. `PYTHONPATH=/Users/nicholas/Projects/glmax/.worktrees/glmax-fit-pipeline-redesign/src pytest -p no:capture tests/test_glm.py` -> `23 passed`
2. `PYTHONPATH=/Users/nicholas/Projects/glmax/.worktrees/glmax-fit-pipeline-redesign/src pytest -p no:capture tests/test_fit_api.py` -> `14 passed`
3. `PYTHONPATH=/Users/nicholas/Projects/glmax/.worktrees/glmax-fit-pipeline-redesign/src pytest -p no:capture tests/test_fitters.py` -> `4 passed`
4. `PYTHONPATH=/Users/nicholas/Projects/glmax/.worktrees/glmax-fit-pipeline-redesign/src pytest -p no:capture tests` -> `41 passed`

Each run reports one non-blocking warning from Lineax deprecation (`NormalCG`).

## Human Test Plan

### HT1: Canonical vs Wrapper User Workflow

1. Run the documented quick-start example from `README.rst`.
2. Confirm both `glmax.fit(...)` and `GLM(...).fit(...)` return `GLMState`.
3. Compare `beta`, `se`, and `p` numerically on the same dataset; values should be aligned.

Expected result: canonical and wrapper calls remain behaviorally consistent for valid inputs.

### HT2: Failure-Semantics Usability Check

1. Invoke fit with non-numeric `X` and with non-finite `offset_eta`.
2. Confirm errors are deterministic (`TypeError` for non-numeric, `ValueError` for non-finite).
3. Verify error messaging matches documented wording in `docs/api/glm.md`.

Expected result: documented failure semantics are understandable and match runtime behavior.

### HT3: Migration Guidance Clarity Review

1. Review `README.rst`, `docs/index.md`, and `docs/api/glm.md`.
2. Validate that deprecation checkpoints include trigger, release window, and migration path.
3. Confirm documentation clearly positions `glmax.fit` as canonical and `GLM.fit` as compatibility shim.

Expected result: migration guidance is clear enough for maintainers to enforce shim-removal gates safely.
