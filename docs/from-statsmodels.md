# From statsmodels

This page maps common [statsmodels](https://www.statsmodels.org/stable/index.html) patterns to their glmax equivalents. The two libraries share most of the same statistical concepts — GLM families, link functions, offsets, robust standard errors — but differ in API shape.

The key structural difference: statsmodels bundles fitting, inference, and prediction into a single result object. glmax separates them into explicit verbs that return explicit nouns. There's no `.summary()` method; inference lives in [`glmax.infer`](api/infer/index.md).

---

## Fitting a GLM

statsmodels exposes a mix of family-specific constructors (`sm.OLS`, `sm.Logit`, `sm.Poisson`) and a generic `sm.GLM`. glmax uses one entry point for everything — the family is an argument, not part of the class name.

**statsmodels:**

```python
import statsmodels.api as sm
import numpy as np

X = sm.add_constant(X_raw)   # statsmodels adds the intercept for you

result = sm.OLS(y, X).fit()                                      # Gaussian
result = sm.Logit(y, X).fit()                                    # Binomial / logit
result = sm.Poisson(y, X).fit()                                  # Poisson
result = sm.GLM(y, X, family=sm.families.Gamma()).fit()          # Gamma
result = sm.NegativeBinomial(y, X).fit()                         # Negative Binomial
```

**glmax:**

```python
import jax.numpy as jnp
import glmax

# glmax does not add an intercept automatically — include it in X
X = jnp.column_stack([jnp.ones(n), X_raw])

fitted = glmax.fit(glmax.Gaussian(), X, y)
fitted = glmax.fit(glmax.Binomial(), X, y)
fitted = glmax.fit(glmax.Poisson(), X, y)
fitted = glmax.fit(glmax.Gamma(), X, y)
fitted = glmax.fit(glmax.NegativeBinomial(), X, y)
```

Fitting returns a [`FittedGLM`](api/fit/index.md) noun. Inference is a separate step:

```python
result = glmax.infer(fitted)
result.params.beta   # coefficients  (cf. result.params)
result.se            # standard errors  (cf. result.bse)
result.stat          # test statistics  (cf. result.tvalues)
result.p             # p-values  (cf. result.pvalues)

fitted.mu            # fitted means  (cf. result.fittedvalues)
glmax.predict(fitted.family, fitted.params, X_new)   # out-of-sample  (cf. result.predict)
```

For Negative Binomial the overdispersion parameter lives in `fitted.params.aux` rather than appended to the coefficient vector.

---

## Overriding the link function

**statsmodels:**

```python
result = sm.GLM(
    y, X,
    family=sm.families.Binomial(sm.families.links.Probit())
).fit()
```

**glmax:**

```python
fitted = glmax.fit(glmax.Binomial(glmax.ProbitLink()), X, y)
```

The link is passed directly to the family constructor. See [Families & Links](api/families-and-links.md) for all supported combinations.

---

## Offsets

Both libraries use `offset=` as a keyword argument. The offset is added to the linear predictor before the inverse link is applied.

**statsmodels:**

```python
result = sm.Poisson(y, X).fit(offset=np.log(exposure))
```

**glmax:**

```python
fitted = glmax.fit(glmax.Poisson(), X, y, offset=jnp.log(exposure))
```

---

## Robust (sandwich) standard errors

**statsmodels:**

```python
result = sm.Poisson(y, X).fit(cov_type="HC3")
```

**glmax:**

```python
result = glmax.infer(fitted, stderr=glmax.HuberError())
```

---

## Residuals

**statsmodels:**

```python
result.resid_pearson    # Pearson residuals
result.resid_deviance   # deviance residuals
```

**glmax:**

```python
pearson  = glmax.check(fitted, diagnostic=glmax.PearsonResidual())
deviance = glmax.check(fitted, diagnostic=glmax.DevianceResidual())
quantile = glmax.check(fitted, diagnostic=glmax.QuantileResidual())
```

Quantile residuals are also available — they're randomised probability integral transform residuals and are especially useful for discrete families like Poisson and Binomial, where Pearson and deviance residuals don't follow a clean reference distribution.

---

## Goodness-of-fit statistics

**statsmodels:**

```python
result.deviance          # residual deviance
result.pearson_chi2      # Pearson χ²
result.aic               # AIC
```

**glmax:**

```python
gof = glmax.check(fitted, diagnostic=glmax.GoodnessOfFit())
gof.deviance       # residual deviance
gof.pearson_chi2   # Pearson χ²
gof.aic            # AIC
```

---

## What glmax doesn't have (yet)

A few things statsmodels provides that glmax doesn't yet support:

- **`.summary()` formatted tables** — inference results are structured arrays, not formatted text. Use `result.params.beta`, `result.se`, etc. directly.
- **Dispersion inference** — standard errors on the dispersion parameter for Gaussian and Gamma are on the roadmap.
- **Per-sample weights** — `weights=` is accepted but not yet implemented.

---

## What glmax adds

Beyond the API differences, glmax is built on JAX, which gives you:

- **Batched fitting** — fit the same model to hundreds of response vectors simultaneously with `eqx.filter_vmap`.
- **Gradients through `fit`** — `jax.grad` and `jax.jvp` work through the fit itself via an Implicit Function Theorem-based custom derivative rule.
- **JIT compilation** — all verbs compile on first call and run fast on CPU, GPU, or TPU thereafter.

See [JAX Transformations](jax-transformations.md) for examples.
