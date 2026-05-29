# Model Fitting

`glmax.fit(family, X, y)` takes a family and observed data arrays and returns a
fitted noun. The fitting strategy is an explicit `fitter=` argument — default
`IRLSFitter`, or `NewtonFitter` for Fisher scoring Newton with backtracking line
search — that can be swapped without changing anything else in the workflow.

Frequency weights use the explicit weight constructor:

```python
fitted = glmax.fit(family, X, y, weights=glmax.weights(freq=w))
```

`freq[i]` means row `i` represents repeated observations. Variance weights and
combined frequency/variance weights are intentionally reserved until their
likelihood, dispersion, and diagnostic semantics are designed.

::: glmax.fit

---

::: glmax.weights

---

::: glmax.Params

---

::: glmax.FitResult

---

::: glmax.FittedGLM
