# glmax

`glmax` provides generalized linear modeling with a grammar-first API.

The canonical workflow is:

1. `specify(...)` a model.
2. `fit(model, data, ...)` to estimate parameters.
3. `predict(model, params, data)` for fitted means.
4. `infer(model, fit_result)` for inferential summaries.
5. `check(model, fit_result)` for diagnostics.

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

fit_result = glmax.fit(model, data)
pred = glmax.predict(model, fit_result.params, data)
infer_result = glmax.infer(model, fit_result)
diagnostics = glmax.check(model, fit_result)
```

## Testing

Use the repository-standard pytest invocation:

```bash
pytest -p no:capture tests
```

## Support

- Issues: https://github.com/mancusolab/glmax/issues
- Source: https://github.com/mancusolab/glmax

## License

`glmax` is distributed under the MIT license.
