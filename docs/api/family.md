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

::: glmax.family.Gaussian

::: glmax.family.Gamma

::: glmax.family.Poisson

::: glmax.family.Binomial

::: glmax.family.NegativeBinomial

## Links

::: glmax.family.AbstractLink

::: glmax.family.IdentityLink

::: glmax.family.LogLink

::: glmax.family.LogitLink

::: glmax.family.InverseLink

::: glmax.family.NBLink

::: glmax.family.PowerLink
