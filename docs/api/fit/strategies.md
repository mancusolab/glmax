# Fitting strategies

`glmax.fit` defaults to `IRLSFitter`. Pass `fitter=` to swap strategies without
changing anything else in the workflow.

```python
fitted_irls   = glmax.fit(family, X, y)
fitted_newton = glmax.fit(family, X, y, fitter=glmax.NewtonFitter())
```

Both strategies return the same `FittedGLM` noun and are `eqx.filter_jit`-compatible.

## Fitter strategies

??? abstract "`glmax.AbstractFitter`"

    ::: glmax.AbstractFitter
        options:
            members:
                - fit

::: glmax.IRLSFitter
    options:
        members:
            - __init__

---

::: glmax.NewtonFitter
    options:
        members:
            - __init__

## Linear solvers

Each fitter delegates its weighted normal-equations step to a `lineax` solver.
Pass any `lx.AbstractLinearSolver` as `solver=` when constructing a fitter.
`lx.Cholesky()` is the default and the fastest option for well-conditioned
systems. Use `lx.QR()` when the design matrix is rank-deficient.

