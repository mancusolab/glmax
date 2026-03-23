# Getting Started

glmax uses four verbs — [`fit`](api/fit/index.md), [`predict`](api/predict.md), [`infer`](api/infer/index.md), and [`check`](api/check.md) — that pass explicit nouns between them. This page walks through each verb in turn.

!!! note "Design philosophy"
    glmax is built around a simple idea: the API should be a grammar of *verbs* that operate on *nouns*. Verbs are plain functions (`fit`, `predict`, `infer`, `check`). Nouns are immutable containers for results (`FittedGLM`, `InferenceResult`, and so on). Nothing is hidden inside a model object. You can inspect, pass, store, or vmap over any noun directly.

If you're coming from statsmodels, the [From statsmodels](from-statsmodels.md) page maps the two APIs side by side.

---

## Fitting a model

You always start with `fit`. It takes a [family](api/families-and-links.md), a design matrix `X`, and a response vector `y`, optimises the negative log-likelihood to convergence, and returns a [`FittedGLM`](api/fit/index.md) that holds everything produced during fitting.

```python
import jax.numpy as jnp
import glmax

X = jnp.array([[1.0, 0.3],
                [1.0, 1.1],
                [1.0, 2.4],
                [1.0, 3.0]])
y = jnp.array([0.5, 1.8, 2.3, 3.1])

fitted = glmax.fit(glmax.Gaussian(), X, y)
```

You can access fit artifacts directly from the returned noun:

```python
fitted.params.beta     # coefficient vector, shape (p,)
fitted.params.disp     # dispersion φ
fitted.mu              # fitted means E[y | X], shape (n,)
fitted.eta             # linear predictor Xβ, shape (n,)
fitted.converged       # True if the fitter converged within tolerance
fitted.num_iters       # number of iterations taken
```

!!! warning "No automatic intercept"
    glmax doesn't add an intercept automatically. Include a column of ones in `X` if you want one, as in the example above. This is intentional — it keeps the design matrix explicit and avoids surprises when you're controlling exactly which covariates are included.

---

## Choosing a family and link

The family encodes the response distribution and variance function. The link function maps the linear predictor to the mean. glmax gives every family a sensible default link, but you can override it.

```python
# Poisson regression — log link by default
fitted = glmax.fit(glmax.Poisson(), X_count, y_count)

# Binomial with probit link instead of the default logit
fitted = glmax.fit(glmax.Binomial(glmax.ProbitLink()), X_bin, y_bin)

# Gamma regression — inverse link by default
fitted = glmax.fit(glmax.Gamma(), X_pos, y_pos)

# Negative Binomial — overdispersion α is estimated and stored in fitted.params.aux
fitted = glmax.fit(glmax.NegativeBinomial(), X_count, y_count)
alpha = fitted.params.aux
```

See [Families & Links](api/families-and-links.md) for the full list.

---

## Making predictions

Once you have a fitted model, `predict` applies it to any design matrix. It doesn't need the full [`FittedGLM`](api/fit/index.md) — just the family and the parameters — so you can also use it for prediction with hand-constructed coefficients or from warm-starting experiments.

```python
# In-sample fitted means (same as fitted.mu)
mu_hat = glmax.predict(fitted.family, fitted.params, X)

# Out-of-sample predictions
X_new = jnp.array([[1.0, 1.5], [1.0, 2.0]])
mu_new = glmax.predict(fitted.family, fitted.params, X_new)
```

Pass `offset` if your model has an exposure or other additive term in the linear predictor:

```python
mu_new = glmax.predict(fitted.family, fitted.params, X_new, offset=log_exposure_new)
```

---

## Inference on coefficients

`infer` takes the fitted noun and returns [`InferenceResult`](api/infer/index.md): coefficient estimates, standard errors, test statistics, and p-values. No refitting happens.

```python
result = glmax.infer(fitted)

result.params.beta   # same as fitted.params.beta
result.se            # standard errors, shape (p,)
result.stat          # test statistics, shape (p,)
result.p             # two-sided p-values, shape (p,)
```

The default is a Wald test with Fisher information standard errors. You can swap either component independently:

```python
# Score test with sandwich (Huber) standard errors
result = glmax.infer(
    fitted,
    inferrer=glmax.ScoreTest(),
    stderr=glmax.HuberError(),
)
```

Huber standard errors are useful when you're uncertain about the variance function or have overdispersion you don't want to model explicitly. See [Inference strategies](api/infer/strategies.md) for the full set of options.

---

## Diagnosing the fit

`check` applies a diagnostic to the fitted noun and returns a typed result. You choose the diagnostic explicitly rather than getting a bundle of everything at once.

Residual diagnostics return an array of the same shape as `y`:

```python
pearson  = glmax.check(fitted, diagnostic=glmax.PearsonResidual())
deviance = glmax.check(fitted, diagnostic=glmax.DevianceResidual())
quantile = glmax.check(fitted, diagnostic=glmax.QuantileResidual())
```

Quantile residuals are randomised probability integral transform residuals — they're the right choice for discrete families like Poisson and Binomial, where simpler residuals don't follow a clean reference distribution.

[`GoodnessOfFit`](api/check.md) and [`Influence`](api/check.md) return structured result nouns:

```python
gof = glmax.check(fitted, diagnostic=glmax.GoodnessOfFit())
gof.pearson_chi2    # Pearson χ² statistic
gof.deviance        # residual deviance
gof.df_resid        # residual degrees of freedom
gof.aic             # Akaike information criterion

influence = glmax.check(fitted, diagnostic=glmax.Influence())
influence.hat        # leverage values (diagonal of hat matrix)
influence.cooks_d    # Cook's distance
influence.dffits     # DFFITS
```

---

## Switching fit strategies

`fit` defaults to [`IRLSFitter`](api/fit/strategies.md), which solves a sequence of weighted least-squares problems. For problems where IRLS overshoots — non-canonical links, near-boundary means — [`NewtonFitter`](api/fit/strategies.md) may converge more reliably. It uses a backtracking Armijo line search to control step size.

```python
# Fisher scoring Newton with automatic line search
fitted = glmax.fit(glmax.Poisson(), X, y, fitter=glmax.NewtonFitter())
```

Both strategies return the same [`FittedGLM`](api/fit/index.md) noun. You can also tune tolerances or swap the underlying linear solver:

```python
import lineax as lx

fitter = glmax.IRLSFitter(solver=lx.QR(), tol=1e-6, max_iter=500)
fitted = glmax.fit(glmax.Gamma(), X, y, fitter=fitter)
```

The default Cholesky solver is fastest for small-to-medium problems. QR handles rank-deficient designs more gracefully.

---

## Offsets and warm-starting

An offset is a fixed term added to the linear predictor before the inverse link. The classic use case is rate modeling in Poisson regression, where the offset is the log of exposure time or population size.

```python
import jax.numpy as jnp

# log(exposure) added to the linear predictor: log(μ) = Xβ + offset
fitted = glmax.fit(glmax.Poisson(), X, y, offset=jnp.log(exposure))
```

Warm-starting lets you seed the solver with parameters from a previous fit. This is useful when refitting the same model on updated data, or when you want to continue from a partially converged solution.

```python
# First fit
fitted = glmax.fit(glmax.Poisson(), X, y)

# Refit on new data, starting from the previous solution
fitted2 = glmax.fit(glmax.Poisson(), X, y_new, init=fitted.params)
```

---

## JAX transformations

`fit`, `predict`, `infer`, and `check` are all JIT-compiled by default and compatible with JAX transforms. See [JAX Transformations](jax-transformations.md) for batched fitting, gradients through the fit, and other transform patterns.
