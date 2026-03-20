# Fitting Strategies

Most users should stay on the public workflow:

```python
fitted = glmax.fit(model, data)
```

This page documents the advanced strategy objects behind that verb. These
helpers live under `glmax` because they are secondary to the
grammar-first package-root API.

## Fitter Strategies

??? abstract "glmax.AbstractFitter"

    ::: glmax.AbstractFitter

::: glmax.IRLSFitter

## Linear Solvers

::: glmax.AbstractLinearSolver

::: glmax.CholeskySolver

::: glmax.QRSolver

::: glmax.CGSolver
