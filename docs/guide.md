# Guide

## Basic workflow

```python
import jax.numpy as jnp
import glmax

X = jnp.array([[1.0, 0.3],
                [1.0, 1.1],
                [1.0, 2.4],
                [1.0, 3.0]])
y = jnp.array([0.5, 1.8, 2.3, 3.1])

fitted   = glmax.fit(glmax.Gaussian(), X, y)
pred     = glmax.predict(fitted.family, fitted.params, X)
inferred = glmax.infer(fitted)
residuals = glmax.check(fitted, diagnostic=glmax.PearsonResidual())
```

`fit` returns a [`FittedGLM`](api/fit/index.md) noun that binds the family and
all fit artifacts. `predict`, `infer`, and `check` all accept it directly —
nothing is refitted.

Access fitted artifacts from the noun:

```python
fitted.params.beta     # coefficient vector
fitted.params.disp     # dispersion
fitted.mu              # fitted means
fitted.eta             # linear predictor
fitted.converged       # bool
fitted.num_iters       # iteration count
```

---

## Families and links

Pass a family instance as the first argument to `fit`. Each family has a
default link; pass an explicit link to override it.

```python
# Poisson with default log link
fitted = glmax.fit(glmax.Poisson(), X_count, y_count)

# Binomial with probit link instead of logit
fitted = glmax.fit(glmax.Binomial(glmax.ProbitLink()), X_bin, y_bin)

# Negative Binomial — overdispersion alpha stored in fitted.params.aux
fitted = glmax.fit(glmax.NegativeBinomial(), X_count, y_count)
```

See [Families & Links](api/families-and-links.md) for the full list of
families and links.

---

## Inference

`infer` returns coefficient-level summaries without refitting. The default
is a Wald test with Fisher information standard errors.

```python
result = glmax.infer(fitted)
result.params   # beta estimates
result.se       # standard errors
result.stat     # test statistics
result.p        # p-values
```

Swap the test or standard error estimator explicitly:

```python
# Score test with sandwich (Huber) standard errors
result = glmax.infer(
    fitted,
    inferrer=glmax.ScoreTest(),
    stderr=glmax.HuberError(),
)
```

---

## Diagnostics

`check` dispatches on the diagnostic type and returns a typed result.

```python
pearson   = glmax.check(fitted, diagnostic=glmax.PearsonResidual())
deviance  = glmax.check(fitted, diagnostic=glmax.DevianceResidual())
quantile  = glmax.check(fitted, diagnostic=glmax.QuantileResidual())
gof       = glmax.check(fitted, diagnostic=glmax.GoodnessOfFit())
influence = glmax.check(fitted, diagnostic=glmax.Influence())
```

---

## Fit strategies

`fit` defaults to `IRLSFitter`. Pass `fitter=` to swap strategies without
changing anything else.

```python
# Default IRLS
fitted = glmax.fit(glmax.Poisson(), X, y)

# Fisher scoring Newton with backtracking Armijo line search
fitted = glmax.fit(glmax.Poisson(), X, y, fitter=glmax.NewtonFitter())
```

`NewtonFitter` converges in fewer outer iterations than `IRLSFitter` on
problems where the IRLS fixed step overshoots — non-canonical links and
near-boundary means in particular. Both strategies return the same
`FittedGLM` noun and are `eqx.filter_jit`-compatible.

Control tolerances and solver explicitly:

```python
import lineax as lx

fitter = glmax.IRLSFitter(solver=lx.QR(), tol=1e-6, max_iter=500)
fitted = glmax.fit(glmax.Gamma(), X, y, fitter=fitter)
```

---

## Offsets and warm-starting

```python
import jax.numpy as jnp

# Offset added to the linear predictor before the inverse link
fitted = glmax.fit(glmax.Poisson(), X, y, offset=log_exposure)

# Warm-start from a previous fit
fitted2 = glmax.fit(glmax.Poisson(), X, y_new, init=fitted.params)
```

---

## JAX transformations

### Batched fitting with `filter_vmap`

`fit` is compatible with `eqx.filter_vmap`. A natural use case is fitting
the same model to many response vectors — bootstrapping, permutation testing,
or multiple outcomes with a shared design matrix.

```python
import jax.random as jr
import equinox as eqx

key = jr.key(0)
BATCH, N = 200, 100

X = jnp.column_stack([jnp.ones(N), jr.normal(key, (N,))])
ys = jr.normal(jr.fold_in(key, 1), (BATCH, N))   # 200 response vectors

# Fit all 200 simultaneously
fitted_batch = eqx.filter_vmap(
    lambda y: glmax.fit(glmax.Gaussian(), X, y)
)(ys)

fitted_batch.params.beta.shape   # (200, 2)
fitted_batch.converged.shape     # (200,)
```

You can also vmap over `(X, y)` jointly — useful for cross-validation folds
or simulation studies:

```python
fitted_batch = eqx.filter_vmap(
    lambda X, y: glmax.fit(glmax.Gaussian(), X, y)
)(X_folds, y_folds)
```

### Gradients

`predict` is fully differentiable. Use `jax.grad` to compute how predictions
change with respect to coefficients or the design matrix:

```python
import jax
from glmax import Params

def total_predicted(beta):
    p = Params(beta=beta, disp=fitted.params.disp, aux=fitted.params.aux)
    return glmax.predict(fitted.family, p, X).sum()

dpred_dbeta = jax.grad(total_predicted)(fitted.params.beta)
```

Forward-mode AD (`jax.jvp`) works through the full fit loop. This lets you
compute directional sensitivities of the fitted coefficients to perturbations
in the data or design matrix:

```python
# How does beta change if X shifts in direction v?
v = jnp.ones_like(X)
_, dbeta_dX = jax.jvp(
    lambda X: glmax.fit(glmax.Gaussian(), X, y).params.beta,
    (X,),
    (v,),
)
```

Reverse-mode AD (`jax.grad`) through the fit loop itself is not supported —
`lax.while_loop` does not carry a reverse-mode tape. If you need implicit
differentiation of the fitted solution, that is on the roadmap.
