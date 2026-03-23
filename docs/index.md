# glmax

`glmax` is a JAX-based generalized linear modeling library.

```python
import jax.numpy as jnp
import glmax

X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
y = jnp.array([0.0, 1.0, 1.0, 2.0])

fitted   = glmax.fit(glmax.Poisson(), X, y)
mu_hat   = glmax.predict(fitted.family, fitted.params, X)
result   = glmax.infer(fitted)
residuals = glmax.check(fitted, diagnostic=glmax.PearsonResidual())
```

Four verbs — [`fit`](api/fit/index.md), [`predict`](api/predict.md), [`infer`](api/infer/index.md), and [`check`](api/check.md) — cover the full modeling workflow. Each takes explicit inputs and returns an explicit result. No hidden state is threaded between calls.

See the [Overview](guide.md) to get started, or [From statsmodels](from-statsmodels.md) if you're migrating from statsmodels.

