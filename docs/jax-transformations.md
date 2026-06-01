# JAX Transformations

All four glmax verbs — [`fit`](api/fit/index.md), [`predict`](api/predict.md),
[`infer`](api/infer/index.md), and [`check`](api/check.md) — are
`@eqx.filter_jit`-wrapped at the public boundary and return JAX-compatible
pytrees.

We provide several practical examples below: batched fits, simulation studies,
bootstrap workflows, differentiable preprocessing, and prediction sensitivity.
Use [`infer`](api/infer/index.md) for ordinary GLM standard errors and p-values.

---

## Batched Fitting

Use `eqx.filter_vmap` when fitting many independent models with the same
structure. It is Equinox's vmapped analogue of `jax.vmap`, and it handles pytrees
with non-array leaves such as families, fitters, and fitted result nouns.

```python
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import glmax

key = jr.key(0)
BATCH, N = 200, 100

X = jnp.column_stack([jnp.ones(N), jr.normal(key, (N,))])
ys = jr.normal(jr.fold_in(key, 1), (BATCH, N))

fitted_batch = eqx.filter_vmap(
    lambda y: glmax.fit(glmax.Gaussian(), X, y)
)(ys)

fitted_batch.params.beta.shape   # (200, 2)
fitted_batch.converged.shape     # (200,)
```

The returned [`FittedGLM`](api/fit/index.md) is itself a batched pytree. Every
array field gains a leading batch dimension.

You can also map over `(X, y)` jointly, which is useful for cross-validation
folds or simulation studies:

```python
fitted_folds = eqx.filter_vmap(
    lambda Xi, yi: glmax.fit(glmax.Gaussian(), Xi, yi)
)(X_folds, y_folds)
```

---

## Bootstrap Fits

Bootstrap workflows are another natural `filter_vmap` use case. Each resample
fits the same model to a different row sample, and the batched coefficient array
can be summarized directly.

```python
def bootstrap_betas(key, X, y, B=200):
    n = y.shape[0]
    keys = jr.split(key, B)

    def one_resample(k):
        idx = jr.choice(k, n, shape=(n,), replace=True)
        return glmax.fit(glmax.Poisson(), X[idx], y[idx]).params.beta

    return eqx.filter_vmap(one_resample)(keys)

betas = bootstrap_betas(jr.key(42), X, y)
betas.shape        # (200, p)
betas.std(axis=0)  # bootstrap standard errors
```

---

## Differentiating `predict`

`predict` is differentiable without special handling. It computes the linear
predictor and then asks the family for the response-scale mean. For most
families this is the inverse link $g^{-1}(\eta)$; for grouped Binomial it also
converts probability to expected count.

```python
import jax
from glmax import Params

def total_predicted(beta):
    params = Params(beta=beta, disp=fitted.params.disp, aux=fitted.params.aux)
    return glmax.predict(fitted.family, params, X).sum()

dpred_dbeta = jax.grad(total_predicted)(fitted.params.beta)
```

This is a good fit for prediction sensitivity: derivatives with respect to
coefficients, design variables, offsets, or continuous preprocessing parameters.

---

## Differentiating Through `fit`

`fit` supports both forward-mode (`jax.jvp`) and reverse-mode (`jax.grad`)
automatic differentiation. The cleanest use case is a scalar downstream loss
that depends on a fitted model, where the differentiated argument is a continuous
upstream quantity.

```python
def held_out_poisson_nll(scale, x_train, y_train, x_test, y_test):
    X_train = jnp.column_stack([jnp.ones_like(x_train), scale * x_train])
    X_test = jnp.column_stack([jnp.ones_like(x_test), scale * x_test])

    fitted = glmax.fit(glmax.Poisson(), X_train, y_train)
    mu_test = glmax.predict(fitted.family, fitted.params, X_test)
    return jnp.sum(mu_test - y_test * jnp.log(mu_test))

d_loss_dscale = jax.grad(held_out_poisson_nll)(
    1.0,
    x_train,
    y_train,
    x_test,
    y_test,
)
```

Here `scale` is a differentiable preprocessing parameter. The gradient flows
through the training fit, the fitted coefficients, and the held-out prediction
loss.

!!! warning "Do not use AD through `fit` as GLM inference"
    Gradients through `fit` are useful for sensitivity analysis and
    differentiable software built on top of glmax. They are not replacements for
    standard errors, p-values, or diagnostic summaries. Use `infer(...)` and
    `check(...)` for those quantities.

---

## JIT Compilation

All public verbs are already JIT-compiled. If you call `fit` inside a larger
JIT-compiled function, you do not need to wrap `fit` again.

```python
@jax.jit
def pipeline(X, y, X_test):
    fitted = glmax.fit(glmax.Poisson(), X, y)
    return glmax.predict(fitted.family, fitted.params, X_test)
```

!!! note "Tracing and recompilation"
    The first call traces and compiles the computation. Later calls with the same
    array shapes are fast. Changing the family or fitter changes static
    structure and retraces. Changing array shapes also retraces; changing array
    values does not.

---

## Advanced

### How `fit` Differentiates

glmax registers a custom JVP rule based on the [Implicit Function
Theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem). It does not
differentiate through every solver iteration.

At the maximum likelihood estimate, the score is zero. Differentiating that
condition gives

$$
H \, d\hat\beta
  = -\partial_{\text{data}}(\nabla_\beta \ell) \cdot d(\text{data}),
$$

where $H = X^\top W X$ is the Fisher information at the converged fit. glmax
solves this linear system to get the coefficient tangent.

### Data Sensitivities

You can use `jax.jvp` to ask how fitted coefficients move under a small data
perturbation. This is a sensitivity calculation, not ordinary coefficient
inference.

```python
X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
y = jnp.array([1.0, 2.0, 3.0, 4.0])

dy = jnp.array([1.0, 0.0, 0.0, 0.0])

_, dbeta = jax.jvp(
    lambda y_: glmax.fit(glmax.Gaussian(), X, y_).params.beta,
    (y,),
    (dy,),
)
```

For discrete responses, derivatives with respect to `y` should be read as a
continuous relaxation. They can be useful for debugging or influence-style
sensitivity checks, but they are not a standard GLM estimand.

### Failure Modes

!!! warning "Derivative paths assume a stable local fit"
    Automatic differentiation through a fit assumes the local fitted solution is
    well behaved. Gradients are not meaningful if the fit did not converge, the
    design is rank deficient, or the optimum is near a boundary where the local
    linearization is unstable.

    For derivative-heavy workflows, check `fitted.converged`, keep design
    matrices well conditioned, and prefer continuous upstream parameters over
    discrete branching or row-selection logic inside the differentiated
    function.
