# Families and links

A GLM is defined by its response family and link function. Pass a family
instance as the first argument to [`glmax.fit`](fit/index.md#glmax.fit):

```python
import glmax

fitted = glmax.fit(glmax.Poisson(), X, y)               # log link by default
fitted = glmax.fit(glmax.Binomial(glmax.ProbitLink()), X, y)  # explicit link
```

The family determines how the linear predictor $\eta = X\beta$ maps to the
mean response $\mu = \mathrm{E}[Y \mid X]$, how the variance scales with
$\mu$, and how [`glmax.Params`](fit/index.md#glmax.Params) fields are interpreted:

- `disp` is the GLM dispersion $\phi$. Gaussian and Gamma use it as EDM
  dispersion; Poisson, Binomial, and Negative Binomial canonicalize it to `1.0`.
- `aux` carries optional family-specific state. Negative Binomial stores its
  overdispersion `alpha` in `aux` while canonical `disp` remains `1.0`.

!!! warning "Response means are not always inverse-link values"
    For most families, the inverse link is the response mean:
    $\mu = g^{-1}(\eta)$. This is the Gaussian, Poisson, Gamma, Inverse
    Gaussian, and Negative Binomial behavior.

    Grouped Binomial is different. With `Binomial(n_trials=N)`, the inverse
    link gives the success probability $p = g^{-1}(\eta)$, but the observed
    response is a success count. The response-scale mean is therefore
    $\mu = Np$. `fitted.mu`, `glmax.predict(...)`, diagnostics, CDFs, and
    deviance calculations use this response-scale mean.

    The general contract is `fitted.mu == family.response_mean(...)`, and
    `predict(...)` returns that same response-scale mean.

---

## Exponential dispersion families

`ExponentialDispersionFamily` defines the common interface that fitting,
inference, diagnostics, and prediction rely on. Concrete families implement
this contract.

!!! warning "`negloglikelihood` returns contributions"
    `negloglikelihood(y, eta, disp, aux)` returns per-observation negative
    log-likelihood contributions with shape `(n,)`, not a scalar objective.
    Fitters own the reduction: `sum(nll)` for unweighted fits or
    `sum(w * nll)` for weighted fits.

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
                - response_mean
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

::: glmax.InverseGaussian
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

Links connect a family parameter to the linear predictor $\eta$. For most
families that parameter is the response mean $\mu$. For grouped Binomial it is
the success probability $p$, and the family then converts $p$ to the
response-scale mean $\mu = Np$. The abstract link contract documents the
forward link, inverse link, and their derivatives so the family layer and
fitting kernels can work against one interface.

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
