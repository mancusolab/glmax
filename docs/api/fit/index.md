# Model Fitting

`glmax.fit(family, X, y)` takes a family and observed data arrays and returns a
fitted noun. The fitting strategy is an explicit `fitter=` argument — default
`IRLSFitter`, or `NewtonFitter` for Fisher scoring Newton with backtracking line
search — that can be swapped without changing anything else in the workflow.

## Offsets and Exposure

`offset` is a fixed additive term in the linear predictor:

$$
\eta = X\beta + \mathrm{offset}.
$$

For Poisson and Negative Binomial models with a log link, exposure is represented
as an offset on the log scale:

```python
fitted = glmax.fit(glmax.Poisson(), X, y, offset=jnp.log(exposure))
```

This models expected counts as
$\mu = \mathrm{exposure} \cdot \exp(X\beta)$.

!!! warning "Pass log exposure, not raw exposure"
    `offset` is added directly to $\eta$. For exposure time, area, population at
    risk, sequencing depth, or similar denominators, pass `jnp.log(exposure)`.
    Raw exposure belongs in the mean model only after taking the log.

::: glmax.fit

---

::: glmax.Params

---

::: glmax.FitResult

---

::: glmax.FittedGLM
