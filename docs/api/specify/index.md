# Families

A GLM is defined by its response family and link function. Pass a family
instance directly to [`glmax.fit`][] as the first argument:

```python
import glmax
from glmax.family import Poisson

fitted = glmax.fit(Poisson(), X, y)
```

The family determines how the linear predictor $\eta = X\beta$ maps to the
mean response $\mu$, and how the variance scales with $\mu$. It also governs
how [`glmax.Params`][] fields are interpreted:

- `disp` is the GLM dispersion / $\phi$. Gaussian and Gamma use it as EDM
  dispersion; Poisson, Binomial, and Negative Binomial canonicalize it to `1.0`.
- `aux` carries optional family-specific state. Negative Binomial stores its
  overdispersion `alpha` in `aux` while canonical `disp` remains `1.0`.

See [Families & Links](families-and-links.md) for the full family and link
reference.
