# Model Specification

`glmax.specify(...)` is the entry point for model specification. Its job is to
define the statistical family and link structure before any data are fit. The
high-level philosophy is that model specification should stay pure: `specify`
returns a noun that describes the model, not a stateful object that already
knows about coefficients, solvers, or observed data.

::: glmax.specify

---

`specify` produces the canonical model noun used by the rest of the workflow.

::: glmax.GLM

