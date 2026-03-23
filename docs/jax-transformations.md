# JAX Transformations

All four glmax verbs — [`fit`](api/fit/index.md), [`predict`](api/predict.md), [`infer`](api/infer/index.md), and [`check`](api/check.md) — are `@eqx.filter_jit`-wrapped at the public boundary and return JAX-compatible pytrees. This means the full suite of JAX transforms works on them.

---

## Batched fitting with `filter_vmap`

`eqx.filter_vmap` is equinox's vmapped analogue of `jax.vmap`, extended to handle pytrees and non-array leaves. A natural use case is fitting the same model to many response vectors simultaneously — bootstrapping, permutation testing, or multiple phenotypes with a shared design matrix.

```python
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import glmax

key = jr.key(0)
BATCH, N = 200, 100

X = jnp.column_stack([jnp.ones(N), jr.normal(key, (N,))])
ys = jr.normal(jr.fold_in(key, 1), (BATCH, N))   # 200 response vectors

fitted_batch = eqx.filter_vmap(
    lambda y: glmax.fit(glmax.Gaussian(), X, y)
)(ys)

fitted_batch.params.beta.shape   # (200, 2)
fitted_batch.converged.shape     # (200,)
```

All 200 fits run in parallel. The returned [`FittedGLM`](api/fit/index.md) is itself a batched pytree: every array field gains a leading batch dimension.

You can also vmap over `(X, y)` jointly, which is useful for cross-validation folds or simulation studies:

```python
fitted_folds = eqx.filter_vmap(
    lambda Xi, yi: glmax.fit(glmax.Gaussian(), Xi, yi)
)(X_folds, y_folds)
```

---

## Differentiating through `fit`

`fit` is a JAX-differentiable primitive. Both forward-mode (`jax.jvp`) and reverse-mode (`jax.grad`) work.

Under the hood, glmax registers a custom JVP rule based on the [Implicit Function Theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem). Rather than differentiating through the solver iterations, glmax computes the tangent analytically: at the MLE the score is zero, and differentiating that condition gives $H \, d\hat\beta = -\partial_{\text{data}}(\nabla_\beta \ell) \cdot d(\text{data})$, where $H = X^\top W X$ is the Fisher information at the converged fit. This avoids accumulating gradients through potentially hundreds of solver steps.

### Forward-mode: sensitivities of β to data perturbations

Use `jax.jvp` when you want directional derivatives — for example, how $\hat\beta$ shifts if the response vector moves in some direction:

```python
import jax

X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
y = jnp.array([1.0, 2.0, 3.0, 4.0])

# Direction: perturb the first observation
dy = jnp.array([1.0, 0.0, 0.0, 0.0])

_, dbeta = jax.jvp(
    lambda y_: glmax.fit(glmax.Gaussian(), X, y_).params.beta,
    (y,),
    (dy,),
)
# dbeta[i] = ∂β̂ᵢ / ∂y₁  (sensitivity of each coefficient to y₁)
```

### Reverse-mode: gradients of a scalar loss

Use `jax.grad` when you have a scalar loss that depends on the fitted coefficients — for instance, a held-out log-likelihood or a regularisation term:

```python
def held_out_loglik(y_train, X_train, y_test, X_test):
    fitted = glmax.fit(glmax.Poisson(), X_train, y_train)
    mu_test = glmax.predict(fitted.family, fitted.params, X_test)
    return -jnp.sum(y_test * jnp.log(mu_test) - mu_test)   # Poisson NLL

# Differentiate w.r.t. y_train and X_train (argnums 0 and 1)
grad_fn = jax.grad(held_out_loglik, argnums=(0, 1))
g_y, g_X = grad_fn(y_train, X_train, y_test, X_test)
```

`jax.value_and_grad` also works:

```python
loss, (g_y, g_X) = jax.value_and_grad(
    held_out_loglik, argnums=(0, 1)
)(y_train, X_train, y_test, X_test)
```

### Differentiating `predict`

`predict` is fully differentiable without any special handling — it's just a matrix multiply and a link inverse. Gradients with respect to `beta` or `X` work out of the box:

```python
from glmax import Params

def total_predicted(beta):
    params = Params(beta=beta, disp=fitted.params.disp, aux=fitted.params.aux)
    return glmax.predict(fitted.family, params, X).sum()

dpred_dbeta = jax.grad(total_predicted)(fitted.params.beta)
```

---

## JIT compilation

All verbs are JIT-compiled on first call. The family instance and fitter strategy are treated as static structure — changing them forces a retrace, while changing `X`, `y`, or other arrays does not.

!!! note "First-call latency"
    The first call to any verb traces and compiles the computation. This can take a few seconds on the first run but is a one-time cost — subsequent calls with the same array shapes are fast. If you're benchmarking, always time the second call.

If you're calling `fit` inside a larger JIT-compiled function, there's no need to double-wrap:

```python
@jax.jit
def pipeline(X, y, X_test):
    fitted = glmax.fit(glmax.Poisson(), X, y)
    return glmax.predict(fitted.family, fitted.params, X_test)
```

The first call traces and compiles. Subsequent calls with the same shapes are fast.

---

## Combining transforms

Transforms compose. For example, to compute bootstrap standard errors by vmapping over resampled response vectors and then taking gradients:

```python
import jax.random as jr

def bootstrap_betas(key, X, y, B=200):
    n = y.shape[0]
    keys = jr.split(key, B)

    def one_resample(k):
        idx = jr.choice(k, n, shape=(n,), replace=True)
        return glmax.fit(glmax.Poisson(), X[idx], y[idx]).params.beta

    return eqx.filter_vmap(one_resample)(keys)

betas = bootstrap_betas(jr.key(42), X, y)
betas.shape       # (200, p)
betas.std(axis=0) # bootstrap standard errors
```
