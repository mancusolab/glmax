# Model Fitting

`glmax.fit(...)` takes a model specification and observed data and returns a
fitted noun. The high-level philosophy is that fitting is a verb, not a model
method: the model stays a pure specification, while the fitting strategy is an
explicit argument that can be swapped without changing the user-facing grammar.

::: glmax.fit

---

`fit` is where the workflow first introduces the data noun, the optional
parameter warm start, and the fitted-model contract returned downstream.


::: glmax.GLMData

---

::: glmax.Params

---

::: glmax.FitResult

---

::: glmax.FittedGLM

