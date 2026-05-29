# Model Prediction

`glmax.predict(...)` applies a model specification and fitted parameters to
data and returns response-scale mean predictions. The high-level philosophy is
that prediction should stay explicit about both the model and the parameter
carrier rather than hiding state inside a fitted object method.

!!! note "Prediction returns means on the response scale"
    `predict(...)` returns $\mathrm{E}[Y \mid X]$ on the same scale as the
    response vector `y` passed to `fit(...)`. For most families this is exactly
    the inverse link $g^{-1}(\eta)$. For grouped-response families there can be
    one extra family-specific conversion.

    For example, `Binomial(n_trials=10)` models success counts. The inverse
    logit gives the success probability $p = g^{-1}(\eta)$, while
    `predict(...)` returns the expected count $\mu = 10p$. If you want
    probabilities from grouped Binomial predictions, divide by `n_trials`.

::: glmax.predict
