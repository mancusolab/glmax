# Human Test Plan: GLM Fit API

Implementation plan: `docs/implementation-plans/2026-03-04-glm-fit-api/`

## Preconditions

- Use repository root `/Users/nicholas/Projects/glmax/.worktrees/glm-fit-api`.
- Ensure Python environment matches project dependencies.

## Automated Checks (already executed)

1. `mkdocs build`
2. `pytest -p no:capture tests/test_glm.py`
3. `pytest -p no:capture tests`

All three commands passed locally.

## Human Verification Checklist

1. Preferred entrypoint messaging
- Open `docs/api/glm.md`, `docs/index.md`, and `README.rst`.
- Confirm examples use `import glmax as gx; gx.fit(...)` as the recommended path.

2. Migration clarity for legacy callers
- Confirm `GLM.fit(...)` compatibility is documented.
- Confirm parameter mapping is explicit:
  - `offset_eta -> offset`
  - `se_estimator -> covariance`
  - solver/tolerance/iteration settings via `options`

3. Deprecation direction messaging
- Confirm docs state `GLM.fit(...)` is compatibility-oriented.
- Confirm warning behavior is described as opt-in (`GLMAX_WARN_GLM_FIT_COMPAT=1`).

4. API contract consistency spot-check
- In a Python REPL, run:
  - `state = gx.fit(gx.GLM(family=gx.Gaussian()), X, y)`
- Confirm expected fields exist on `state`:
  - `beta`, `se`, `z`, `p`, `eta`, `mu`, `glm_wt`, `num_iters`, `converged`, `infor_inv`, `resid`, `alpha`.

## Pass Criteria

- All automated checks pass.
- All checklist items above are confirmed without contradictions.
