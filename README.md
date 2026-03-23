# glmax

!!! warning

    THIS PROJECT IS STILL IN ALPHA AND SUBJEC TO CHANGE DRAMATICALLY; USE AT YOUR OWN RISK**

 `glmax` provides generalized linear modeling with a grammar-first API built
around explicit nouns and top-level verbs.

The canonical workflow is:

1. `fit(family, X, y, *, offset=None, init=None, fitter=...) -> FittedGLM`
2. `predict(family, params, X, *, offset=None)` for fitted means
3. `infer(fitted, inferrer=None, stderr=...)` for inferential summaries without refitting
4. `check(fitted)` for diagnostics without refitting

The canonical nouns are `Params`, `FitResult`, `FittedGLM`, `InferenceResult`, `AbstractDiagnostic`, `GofStats`, and `InfluenceStats`.
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

See [docs/index.md](docs/index.md) for the workflow overview and the
[Families & Links guide](docs/api/specify/families-and-links.md) for the
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
from glmax.family import Gaussian

X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
y = jnp.array([0.1, 1.2, 1.9, 3.1])

fitted = glmax.fit(Gaussian(), X, y)
pred = glmax.predict(fitted.family, fitted.params, X)
infer_result = glmax.infer(fitted)
# Route through an explicit inferrer when needed.
score_result = glmax.infer(fitted, inferrer=glmax.ScoreTest())
pearson = glmax.check(fitted)
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
