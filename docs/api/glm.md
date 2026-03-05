# GLM API

## Canonical Workflow

Use `glmax.fit` as the canonical public entrypoint for fitting. It owns:

- boundary normalization (`jnp.asarray`, dtype/rank/shape/finiteness checks)
- fitter dispatch (`IRLSFitter` by default, custom fitter injection supported)
- covariance/test strategy composition for returned `GLMState`

`GLM.fit` is a compatibility wrapper that delegates to the same canonical path.
Valid-input outputs are expected to remain aligned between direct and wrapper usage.

## Compatibility Guarantees and Deprecation Checkpoints

- Current guarantee: `GLM.fit` delegates to `glmax.fit`, sharing boundary
  normalization and fitter validation semantics.
- Deprecation trigger: parity and boundary-regression tests remain stable while
  canonical usage guidance is published in user-facing docs.
- Minimum release window: retain `GLM.fit` for at least two minor releases after
  an explicit deprecation notice.
- Migration guidance: call `glmax.fit(...)` directly with explicit `family`,
  `solver`, and optional `fitter` arguments.

## Failure Semantics

Boundary failures are deterministic and occur before numerics execution:

- `TypeError`: non-numeric `X`, `y`, `offset_eta`, `init`, `alpha_init`, or invalid fitter type
- `ValueError`: invalid rank/shape constraints
- `ValueError`: non-finite boundary values (NaN/Inf)

These semantics are regression-tested in `tests/test_fit_api.py`, `tests/test_fitters.py`,
and `tests/test_glm.py`.

## Reference

### Canonical Fit

::: glmax.fit

### GLM Model

::: glmax.GLM
