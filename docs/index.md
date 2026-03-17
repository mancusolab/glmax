# glmax

`glmax` is a JAX-based generalized linear modeling library with a grammar-first API.

## Workflow

Five canonical verbs form the complete user workflow:

```python
import jax.numpy as jnp
import glmax
from glmax import GLMData
from glmax.family import Poisson

model = glmax.specify(family=Poisson())
data  = GLMData(
    X=jnp.array([[0.0], [1.0], [2.0], [3.0]]),
    y=jnp.array([0.0,   1.0,   1.0,   2.0]),
)

fitted = glmax.fit(model, data)
mu_hat = glmax.predict(model, fitted.params, data)
result = glmax.infer(fitted)
diag   = glmax.check(fitted)
```

Each verb takes and returns explicit nouns. No hidden state is threaded between calls.

## API surface

**Verbs** — [`specify`](api/verbs.md), [`fit`](api/verbs.md), [`predict`](api/verbs.md), [`infer`](api/verbs.md), [`check`](api/verbs.md)

**Nouns** — [`GLMData`](api/nouns.md), [`GLM`](api/nouns.md), [`Params`](api/nouns.md), [`FitResult`](api/nouns.md), [`FittedGLM`](api/nouns.md), [`InferenceResult`](api/nouns.md), [`Diagnostics`](api/nouns.md)

**Fitting** — [`AbstractFitter`](api/fitters.md), [`IRLSFitter`](api/fitters.md), [`AbstractLinearSolver`](api/fitters.md), [`CholeskySolver`](api/fitters.md), [`QRSolver`](api/fitters.md), [`CGSolver`](api/fitters.md)

**Inference** — [`AbstractTest`](api/inference.md), [`WaldTest`](api/inference.md), [`ScoreTest`](api/inference.md), [`AbstractStdErrEstimator`](api/inference.md), [`FisherInfoError`](api/inference.md), [`HuberError`](api/inference.md)

**Families** — [`Gaussian`](api/family.md), [`Poisson`](api/family.md), [`Binomial`](api/family.md), [`NegativeBinomial`](api/family.md), [`Gamma`](api/family.md)
