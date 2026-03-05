# GLM Grammar API

This page summarizes the canonical noun/verb contracts for `glmax`.

## Verbs

- `glmax.specify(*, family=None, solver=None) -> GLM`
- `glmax.fit(model, data, init=None, *, fitter=...) -> FitResult`
- `glmax.predict(model, params, data) -> Array`
- `glmax.infer(model, fit_result) -> InferenceResult`
- `glmax.check(model, fit_result) -> Diagnostics`

## Nouns

- `GLMData`
  - `X`: rank-2 design matrix `(n, p)`
  - `y`: rank-1 response vector `(n,)`
  - optional `offset`, `weights`, `mask`
- `Params`
  - `beta`: rank-1 parameter vector `(p,)`
  - `disp`: scalar canonical dispersion
- `FitResult`
  - `params`, `se`, `z`, `p`, `eta`, `mu`, `glm_wt`
  - `diagnostics` (converged, iteration count, objective metadata)
  - `curvature`, `score_residual`
- `InferenceResult`
  - `params`, `se`, `z`, `p`
- `Diagnostics`
  - `converged`, `num_iters`, `objective`, `objective_delta`

## Usage Pattern

```python
import glmax

model = glmax.specify(...)
fit_result = glmax.fit(model, data)
pred = glmax.predict(model, fit_result.params, data)
infer_result = glmax.infer(model, fit_result)
diagnostics = glmax.check(model, fit_result)
```
