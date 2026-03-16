# glmax

`glmax` is a JAX-based generalized linear modeling library with a grammar-first API.

## Canonical API Surface

`glmax` exposes five canonical verbs:

- `specify(...)` creates a `GLM` model
- `fit(model, data, init=None, *, fitter=...)` estimates model parameters and returns a `FittedGLM`
- `predict(model, params, data)` computes predictions
- `infer(fitted, inferrer=None, stderr=...)` computes inferential summaries without refitting
- `check(fitted)` returns diagnostics without refitting

Data and parameter contracts are explicit nouns:

- `GLMData`
- `Params`
- `FitResult`
- `FittedGLM`
- `InferenceResult`
- `Diagnostics`

Package-root inference helpers are also public:

- `AbstractInferrer`
- `WaldInferrer`
- `ScoreInferrer`
- `AbstractStdErrEstimator`
- `FisherInfoError`
- `HuberError`

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

fitted = glmax.fit(model, data)
pred = glmax.predict(model, fitted.params, data)
infer_result = glmax.infer(fitted)
score_result = glmax.infer(fitted, inferrer=glmax.ScoreTest())
diagnostics = glmax.check(fitted)
```

## Verification

Run tests with the repository-standard command used by project scripts:

```bash
pytest -p no:capture tests
```
