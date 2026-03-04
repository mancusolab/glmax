# GLM Fit API

`glmax` now exposes a package-level fit API:

```python
import glmax as gx

model = gx.GLM(family=gx.Gaussian())
state = gx.fit(model, X, y)
```

This is the preferred path for new code.

## Recommended Usage

```python
import glmax as gx

model = gx.GLM(family=gx.Poisson(), solver=gx.QRSolver())
state = gx.fit(
    model,
    X,
    y,
    offset=offset,                 # optional, length n_samples
    covariance=gx.FisherInfoError(),
    options={"max_iter": 1000, "tol": 1e-3, "step_size": 1.0},
)
```

`state` is a `gx.GLMState` with fields:
`beta`, `se`, `p`, `eta`, `mu`, `glm_wt`, `num_iters`, `converged`, `infor_inv`, `resid`, `alpha`.

## Migration From `GLM.fit`

Legacy wrapper usage remains supported:

```python
import glmax as gx

model = gx.GLM(family=gx.Gaussian())
legacy_state = model.fit(X, y, offset_eta=offset)
```

Migration mapping:

- `model.fit(X, y, offset_eta=offset)` -> `gx.fit(model, X, y, offset=offset)`
- `model.fit(..., se_estimator=est)` -> `gx.fit(..., covariance=est)`
- `model.fit(..., max_iter=..., tol=..., step_size=..., alpha_init=...)` -> `gx.fit(..., options={...})`

## Compatibility And Deprecation Direction

- `GLM.fit(...)` currently delegates to `gx.fit(...)` for backward compatibility.
- `GLM.fit(...)` can emit an opt-in compatibility warning when `GLMAX_WARN_GLM_FIT_COMPAT=1`.
- Long-term direction: keep `gx.fit(...)` as the canonical entrypoint for user code.
