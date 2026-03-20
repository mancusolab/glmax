# Inference Strategies

`glmax.infer(fitted)` consumes the fitted noun without refitting. Covariance
scaling uses `fitted.params.disp` as the GLM dispersion source of truth.
Negative Binomial keeps its family-specific `alpha` in `fitted.params.aux`.

## Hypothesis Tests

::: glmax.AbstractTest

::: glmax.WaldTest

::: glmax.ScoreTest

## Standard-Error Estimators

::: glmax.AbstractStdErrEstimator

::: glmax.FisherInfoError

::: glmax.HuberError
