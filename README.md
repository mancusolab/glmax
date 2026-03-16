# glmax

`glmax` provides generalized linear modeling with a grammar-first API.

The canonical workflow is:

1. `specify(...) -> GLM`
2. `fit(model, data, init=None, *, fitter=...) -> FittedGLM`
3. `predict(model, params, data)` for fitted means
4. `infer(fitted, stderr=...)` for inferential summaries without refitting
5. `check(fitted)` for diagnostics without refitting

The canonical nouns are `GLMData`, `Params`, `FitResult`, `FittedGLM`, `InferenceResult`, and `Diagnostics`.

## Installation

```bash
git clone https://github.com/mancusolab/glmax.git
cd glmax
pip install .
```

## Quickstart

```python
import jax.numpy as jnp
import glmax

from glmax import GLMData
from glmax.family import Gaussian

model = glmax.specify(family=Gaussian())
data = GLMData(
    X=jnp.array([[0.0], [1.0], [2.0], [3.0]]),
    y=jnp.array([0.1, 1.2, 1.9, 3.1]),
)

fitted = glmax.fit(model, data)
params = fitted.params
pred = glmax.predict(model, params, data)
infer_result = glmax.infer(fitted)
diagnostics = glmax.check(fitted)
```

## Testing

Use the repository-standard pytest invocation (also wired into project scripts):

```bash
pytest -p no:capture tests
```

## Support

- Issues: https://github.com/mancusolab/glmax/issues
- Source: https://github.com/mancusolab/glmax

## License

`glmax` is distributed under the MIT license.
