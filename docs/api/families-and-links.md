# Families and links

A GLM is defined by its response family and link function. Pass a family
instance as the first argument to [`glmax.fit`][]:

```python
import glmax

fitted = glmax.fit(glmax.Poisson(), X, y)               # log link by default
fitted = glmax.fit(glmax.Binomial(glmax.ProbitLink()), X, y)  # explicit link
```

The family determines how the linear predictor $\eta = X\beta$ maps to the
mean response $\mu$, how the variance scales with $\mu$, and how
[`glmax.Params`][] fields are interpreted:

- `disp` is the GLM dispersion $\phi$. Gaussian and Gamma use it as EDM
  dispersion; Poisson, Binomial, and Negative Binomial canonicalize it to `1.0`.
- `aux` carries optional family-specific state. Negative Binomial stores its
  overdispersion `alpha` in `aux` while canonical `disp` remains `1.0`.

---

## Exponential dispersion families

`ExponentialDispersionFamily` defines the common interface that fitting,
inference, diagnostics, and prediction rely on. Concrete families implement
this contract.

??? abstract "`glmax.ExponentialDispersionFamily`"

    ::: glmax.ExponentialDispersionFamily
        options:
            members:
                - negloglikelihood
                - variance
                - cdf
                - deviance_contribs
                - sample
                - calc_weight
                - init_eta
                - update_nuisance
                - init_nuisance


::: glmax.Gaussian
    options:
        members:
            - __init__

---

::: glmax.Gamma
    options:
        members:
            - __init__

---

::: glmax.Poisson
    options:
        members:
            - __init__

---

::: glmax.Binomial
    options:
        members:
            - __init__

---

::: glmax.NegativeBinomial
    options:
        members:
            - __init__

---

## Link functions

Links connect the mean response $\mu$ to the linear predictor $\eta$. The
abstract link contract documents the forward link, inverse link, and their
derivatives so the family layer and fitting kernels can work against one
interface.

??? abstract "`glmax.AbstractLink`"

    ::: glmax.AbstractLink
        options:
            members:
                - __call__
                - inverse
                - deriv
                - inverse_deriv


::: glmax.IdentityLink
    options:
        members:
            - __init__

---

::: glmax.LogLink
    options:
        members:
            - __init__

---

::: glmax.LogitLink
    options:
        members:
            - __init__

---

::: glmax.InverseLink
    options:
        members:
            - __init__

---

::: glmax.PowerLink
    options:
        members:
            - __init__

---

::: glmax.NBLink
    options:
        members:
            - __init__

---

::: glmax.ProbitLink
    options:
        members:
            - __init__

---

::: glmax.CLogLogLink
    options:
        members:
            - __init__

---

::: glmax.LogLogLink
    options:
        members:
            - __init__

---

::: glmax.SqrtLink
    options:
        members:
            - __init__

---

::: glmax.CauchitLink
    options:
        members:
            - __init__
