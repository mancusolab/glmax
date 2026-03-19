# Nouns

`glmax` passes explicit nouns between the top-level verbs. The shared parameter
carrier is `Params(beta, disp, aux)`.

## Parameter Semantics

| Field | Meaning |
| --- | --- |
| `beta` | Regression coefficients with shape `(p,)`. |
| `disp` | GLM dispersion / `phi`. Gaussian and Gamma store EDM dispersion here. Poisson, Binomial, and Negative Binomial canonicalize it to `1.0`. |
| `aux` | Optional family-specific state. Negative Binomial stores `alpha` here; families without auxiliary state keep `aux is None`. |

## `GLMData`

::: glmax.GLMData

## `GLM`

::: glmax.GLM
    options:
      show_signature: false
      show_signature_annotations: false
      signature_crossrefs: false

## `Params`

::: glmax.Params

## `FitResult`

::: glmax.FitResult

## `FittedGLM`

::: glmax.FittedGLM

## `InferenceResult`

::: glmax.InferenceResult

## `AbstractDiagnostic`

::: glmax.AbstractDiagnostic

## `GofStats`

::: glmax.GofStats

## `InfluenceStats`

::: glmax.InfluenceStats
