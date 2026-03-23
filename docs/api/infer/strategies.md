# Inference strategies and standard errors

`infer` delegates to an explicit hypothesis-test strategy and, when needed, an
explicit covariance estimator. This separation keeps the workflow clear about
what inferential assumptions are being used after fitting.

## Hypothesis-Test Contract

??? abstract "`glmax.AbstractTest`"

    ::: glmax.AbstractTest

## Concrete Tests

::: glmax.WaldTest

---

::: glmax.ScoreTest

## Standard-Error Contract

Covariance estimators are separate strategy objects so the same fitted noun can
be paired with different error models.

??? abstract "`glmax.AbstractStdErrEstimator`"

    ::: glmax.AbstractStdErrEstimator

## Concrete Estimators

::: glmax.FisherInfoError

---

::: glmax.HuberError

