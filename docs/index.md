# glmax

`glmax` is a JAX-based generalized linear modeling library with a grammar-first API.

## Canonical API Surface

`glmax` exposes five canonical verbs:

- `specify(...)` creates a `GLM` model.
- `fit(model, data, ...)` estimates model parameters.
- `predict(model, params, data)` computes predictions.
- `infer(model, fit_result)` computes inferential summaries without refitting.
- `check(model, fit_result)` returns diagnostics without refitting.

Data and parameter contracts are explicit nouns:

- `GLMData`
- `Params`
- `FitResult`
- `InferenceResult`
- `Diagnostics`

## Quick Example

```python
import jax.numpy as jnp
import glmax

from glmax import GLMData
from glmax.family import Poisson

model = glmax.specify(family=Poisson())
data = GLMData(
    X=jnp.array([[0.0], [1.0], [2.0], [3.0]]),
    y=jnp.array([0.0, 1.0, 1.0, 2.0]),
)

fit_result = glmax.fit(model, data)
pred = glmax.predict(model, fit_result.params, data)
infer_result = glmax.infer(model, fit_result)
diagnostics = glmax.check(model, fit_result)
```

## Verification

Run tests with the repository-standard command:

```bash
pytest -p no:capture tests
```
