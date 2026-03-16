# GLM Grammar API

This page summarizes the canonical noun/verb contracts for `glmax`.

## Verbs

- `glmax.specify(*, family=None, solver=None) -> GLM`
- `glmax.fit(model, data, init=None, *, fitter=...) -> FittedGLM`
- `glmax.predict(model, params, data) -> Array`
- `glmax.infer(fitted, stderr=...) -> InferenceResult`
- `glmax.check(fitted) -> Diagnostics`

## Nouns

- `GLMData`
  - `X`: rank-2 design matrix `(n, p)`
  - `y`: rank-1 response vector `(n,)`
  - optional `offset`, `weights`, `mask`
- `Params`
  - `beta`: rank-1 parameter vector `(p,)`
  - `disp`: scalar canonical dispersion
- `FitResult`
  - internal fitter output
  - `params`, `eta`, `mu`, `glm_wt`
  - `X`, `y`
  - `converged`, `num_iters`, `objective`, `objective_delta`
  - `score_residual`
- `FittedGLM`
  - `model`, `result`
  - forwards common fit artifacts like `params`, `eta`, and `mu`
- `InferenceResult`
  - `params`, `se`, `stat`, `p`
  - `stat` is the public test-statistic field; the legacy `.z` attribute is removed
- `Diagnostics`
  - placeholder model-fit diagnostics contract

## Usage Pattern

```python
import glmax

model = glmax.specify(...)
fitted = glmax.fit(model, data)
pred = glmax.predict(model, fitted.params, data)
infer_result = glmax.infer(fitted)
diagnostics = glmax.check(fitted)
```

`glmax.GLM.fit(...)` is not part of the public API contract; use the top-level
grammar workflow shown above.
