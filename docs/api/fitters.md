# Fitting Strategies

Most users should stay on the public workflow:

```python
fitted = glmax.fit(model, data)
```

This page documents the advanced strategy objects behind that verb. These
helpers live under `glmax._fit` because they are secondary to the grammar-first
package-root API.

## Public Entry Point

::: glmax.fit

## Fitter Strategies

::: glmax.AbstractFitter

::: glmax._fit.IRLSFitter

## Linear Solvers

::: glmax._fit.AbstractLinearSolver

::: glmax._fit.CholeskySolver

::: glmax._fit.QRSolver

::: glmax._fit.CGSolver
