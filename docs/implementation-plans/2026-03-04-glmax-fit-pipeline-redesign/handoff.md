# GLMAX Fit Pipeline Redesign Handoff Notes

## Summary

- Canonical fit orchestration is `glmax.fit`.
- `GLM.fit` is retained as a compatibility wrapper and delegates to canonical fit behavior.
- Boundary validation and fitter-contract failures are deterministic and regression-tested.

## What Was Validated

The redesign is validated against these acceptance-criteria targets:

- AC1.2: wrapper/direct parity stays within existing tolerances.
- AC3.2: fitter strategy injection remains supported.
- AC5.2: failure-mode and compatibility semantics are documented and test-referenced.

Regression evidence collected during Phase 6:

- `PYTHONPATH=/Users/nicholas/Projects/glmax/.worktrees/glmax-fit-pipeline-redesign/src pytest -p no:capture tests`
  - Result: `41 passed, 1 warning`.
- `PYTHONPATH=/Users/nicholas/Projects/glmax/.worktrees/glmax-fit-pipeline-redesign/src pytest -p no:capture tests/test_fit_api.py tests/test_fitters.py`
  - Result: `18 passed, 1 warning`.

## Temporary Compatibility Behavior

- `GLM.fit` remains available as a migration-safe wrapper.
- Wrapper calls share canonical boundary normalization, fitter validation, and fit orchestration.
- Compatibility and failure semantics are documented in:
  - `src/glmax/fit.py`
  - `src/glmax/glm.py`
  - `README.rst`
  - `docs/index.md`
  - `docs/api/glm.md`

## Shim Removal Exit Criteria

Before any `GLM.fit` shim removal decision, all of the following must be true:

1. Canonical parity and boundary-regression tests remain passing in CI.
2. A formal deprecation notice has been present for at least two minor releases.
3. User-facing migration guidance for direct `glmax.fit` usage remains current.
4. Risk-log entries related to shim removal are explicitly accepted or mitigated.

## Follow-Up Ownership

- Primary owner for compatibility lifecycle: GLMAX maintainers.
- Risk tracking and checkpoint updates: GLMAX maintainers during release planning.
