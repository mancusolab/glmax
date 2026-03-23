# Model Fitting

`glmax.fit(family, X, y)` takes a family and observed data arrays and returns a
fitted noun. The fitting strategy is an explicit `fitter=` argument — default
`IRLSFitter`, or `NewtonFitter` for Fisher scoring Newton with backtracking line
search — that can be swapped without changing anything else in the workflow.

::: glmax.fit

---

::: glmax.Params

---

::: glmax.FitResult

---

::: glmax.FittedGLM

