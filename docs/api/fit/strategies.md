# Fitting strategies

`glmax.fit` defaults to `IRLSFitter`. Pass `fitter=` to swap strategies without
changing anything else in the workflow.

```python
fitted_irls   = glmax.fit(family, X, y)
fitted_newton = glmax.fit(family, X, y, fitter=glmax.NewtonFitter())
```

Both strategies return the same `FittedGLM` noun and work with JIT compilation and JAX transforms.

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

Each fitter uses a [lineax](https://docs.kidger.site/lineax/) solver for its internal linear system.
Pass any `lineax.AbstractLinearSolver` as `solver=` when constructing a fitter.
`lineax.Cholesky()` is the default and the fastest option for well-conditioned
systems. Use `lineax.QR()` when the design matrix is rank-deficient.

