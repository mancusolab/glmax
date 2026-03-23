# glmax

`glmax` is a JAX-based generalized linear modeling library with a grammar-first API.

## Workflow

Five canonical verbs form the complete user workflow:

```python
import jax.numpy as jnp
import glmax
from glmax.family import Poisson

X = jnp.array([[0.0], [1.0], [2.0], [3.0]])
y = jnp.array([0.0,   1.0,   1.0,   2.0])

fitted = glmax.fit(Poisson(), X, y)
mu_hat = glmax.predict(fitted.family, fitted.params, X)
result = glmax.infer(fitted)
pearson = glmax.check(fitted)
```

Each verb takes and returns explicit nouns. No hidden state is threaded between calls.

## Parameter Carrier

`Params(beta, disp, aux)` is the parameter carrier shared by `fit`, `predict`,
and `infer`.

- `disp` is GLM dispersion / `phi`.
- `aux` is optional family-specific state.
- Negative Binomial stores `alpha` in `aux` while canonical `disp` remains `1.0`.
- Gaussian and Gamma use `disp` as EDM dispersion and ignore `aux`.

## API surface

**Workflow** — [`fit`](api/fit/index.md), [`predict`](api/predict.md), [`infer`](api/infer/index.md), [`check`](api/check.md)

**Families & Links** — [`Families & Links`](api/specify/families-and-links.md)

**Advanced Fitting** — [`Strategies & Solvers`](api/fit/strategies.md)

**Advanced Inference** — [`Strategies & Standard Errors`](api/infer/strategies.md)

The API is organized around the workflow rather than around catch-all noun and
verb inventories. Each verb page explains the role that step plays in the
grammar, then documents the function and the nouns it introduces.
