# Families And Links

The `specify` step chooses a response family and its link function. This is
where the statistical meaning of `Params(beta, disp, aux)` is determined:
different families interpret `disp` and `aux` differently, but the workflow
continues to pass the same explicit nouns between verbs.

---

`ExponentialDispersionFamily` defines the common methods that fitting,
prediction, diagnostics, and inference rely on. Concrete families plug into
the grammar through this contract.

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

## Concrete Families

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

## Link Contract

Links connect the mean response $\mu$ to the linear predictor $\eta$. The
abstract link contract documents the forward link, inverse link, and their
derivatives so the family layer and GLM kernels can work against one interface.

??? abstract "`glmax.AbstractLink`"

    ::: glmax.AbstractLink
        options:
            members:
                - __call__
                - inverse
                - deriv
                - inverse_deriv

## Concrete Links

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

