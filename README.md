# glmax

> **Alpha software** — API may change without notice. Use at your own risk.

Grammar-first generalized linear modeling in JAX. JIT-compiled end-to-end and differentiable through the fitted parameters via the implicit function theorem.

## Example

```python
import jax.numpy as jnp
import glmax

X = jnp.array([[1.0, 0.5], [1.0, -0.3], [1.0, 1.2], [1.0, -0.8]])
y = jnp.array([2.0, 1.0, 4.0, 1.0])

fitted  = glmax.fit(glmax.Poisson(), X, y)
pred    = glmax.predict(fitted.family, fitted.params, X)
result  = glmax.infer(fitted)
diag    = glmax.check(fitted)
```

Four verbs — `fit`, `predict`, `infer`, and `check` — cover the full modeling workflow. Each takes explicit inputs and returns an explicit result. No hidden state is threaded between calls.

See the [docs](https://mancusolab.github.io/glmax) for the full API reference and guides.

## Installation

```bash
pip install git+https://github.com/mancusolab/glmax.git
```


## Performance

Benchmarked against [statsmodels](https://www.statsmodels.org/) on Poisson regression. Timing uses 10 steady-state runs after JIT warm-up.

| n | p | statsmodels (ms) | glmax (ms) | speedup | runtime |
|------:|----:|-----------------:|-----------:|--------:|---------|
| 500 | 10 | 4.32 | 0.92 | 4.7× | CPU |
| 2,000 | 20 | 277.76 | 4.14 | 67.1× | CPU |
| 10,000 | 50 | 1428.76 | 42.77 | 33.4× | CPU |
| 500 | 10 | 2.97 | 3.00 | 1.0× | T4 GPU |
| 2,000 | 20 | 13.94 | 4.38 | 3.2× | T4 GPU |
| 10,000 | 50 | 212.70 | 17.94 | 11.9× | T4 GPU |
| 500 | 10 | 1.90 | 2.89 | 0.7× | v5e-1 TPU |
| 2,000 | 20 | 8.46 | 8.80 | 1.0× | v5e-1 TPU |
| 10,000 | 50 | 1220.66 | 25.65 | 47.6× | v5e-1 TPU |

See [`examples/benchmark_colab.ipynb`](examples/benchmark_colab.ipynb) for the full benchmark notebook.

## Testing

```bash
pytest -p no:capture tests
```

## License

MIT
