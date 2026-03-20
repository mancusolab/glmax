# Families & Links

Families determine how the shared `Params(beta, disp, aux)` carrier is
interpreted.

## Family Parameter Semantics

| Family | `disp` | `aux` |
| --- | --- | --- |
| `Gaussian` | EDM dispersion / variance parameter | ignored |
| `Gamma` | EDM dispersion | ignored |
| `Poisson` | canonical `1.0` | ignored |
| `Binomial` | canonical `1.0` | ignored |
| `NegativeBinomial` | canonical `1.0` | stores auxiliary `alpha` |

## Families

??? abstract "glmax.ExponentialDispersionFamily"

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

---

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


## Links

??? abstract "glmax.AbstractLink"

    ::: glmax.AbstractLink
        options:
            members:
                - __call__
                - inverse
                - deriv
                - inverse_deriv

---

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
