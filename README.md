# glmax

`glmax` provides generalized linear modeling with a grammar-first API built
around explicit nouns and top-level verbs.

The canonical workflow is:

1. `specify(...) -> GLM`
2. `fit(model, data, init=None, *, fitter=...) -> FittedGLM`
3. `predict(model, params, data)` for fitted means
4. `infer(fitted, inferrer=None, stderr=...)` for inferential summaries without refitting
5. `check(fitted)` for diagnostics without refitting

The canonical nouns are `GLMData`, `Params`, `FitResult`, `FittedGLM`, `InferenceResult`, `AbstractDiagnostic`, `GofStats`, and `InfluenceStats`.
The package-root also exports inference strategy and stderr types:
`AbstractTest`, `WaldTest`, `ScoreTest`,
`AbstractStdErrEstimator`, `FisherInfoError`, and `HuberError`.

`Params(beta, disp, aux)` is the shared parameter carrier across `fit`,
`predict`, and `infer`:

- `beta` stores regression coefficients.
- `disp` stores GLM dispersion. Gaussian and Gamma use it as EDM dispersion;
  Poisson, Binomial, and Negative Binomial canonicalize it to `1.0`.
- `aux` stores optional family-specific state. Negative Binomial stores its
  `alpha` here while canonical `disp` remains `1.0`.

See [docs/index.md](docs/index.md) and the [API reference](docs/api/nouns.md)
for the public contract, and the [family guide](docs/api/family.md) for the
family-specific `disp`/`aux` split.

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
# Route through an explicit inferrer when needed.
score_result = glmax.infer(fitted, inferrer=glmax.ScoreTest())
pearson, deviance, quantile, gof, influence = glmax.check(fitted)
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
