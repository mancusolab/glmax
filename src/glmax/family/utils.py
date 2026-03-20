import equinox as eqx
import jax.numpy as jnp

from jax.scipy.special import betainc, expit
from jaxtyping import ArrayLike


def _clipped_expit(x):
    finfo = jnp.finfo(jnp.result_type(x))
    return jnp.clip(expit(x), min=finfo.tiny, max=1.0 - finfo.eps)


def _grad_per_sample(func, x):
    r"""Get a per-sample gradient via `eqx.filter_vmap(eqx.filter_grad(...))`.

    Uses eqx.filter_vmap and eqx.filter_grad instead of raw jax.vmap/jax.grad
    so that eqx.Module dynamic leaves (e.g. PowerLink.power, NBLink.alpha) are
    partitioned correctly and never treated as a vmap batch axis under nested
    vmap transforms.
    """
    return eqx.filter_vmap(eqx.filter_grad(func))(x)


def t_cdf(value: ArrayLike, df: float, loc: ArrayLike = 0.0, scale: ArrayLike = 1.0):
    r"""Evaluate the Student-$t$ cumulative distribution function.

    This uses the Beta-function identity for the Student-$t$ distribution,
    where `df` is the degrees of freedom, `loc` is the location parameter,
    and `scale` is the scale parameter.
    """
    # Ref: https://en.wikipedia.org/wiki/Student's_t-distribution#Related_distributions
    # X^2 ~ F(1, df) -> df / (df + X^2) ~ Beta(df/2, 0.5)
    scaled = (value - loc) / scale
    scaled_squared = scaled * scaled
    beta_value = df / (df + scaled_squared)

    # when scaled < 0, returns 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    # when scaled > 0, returns 1 - 0.5 * Beta(df/2, 0.5).cdf(beta_value)
    return 0.5 * (1 + jnp.sign(scaled) - jnp.sign(scaled) * betainc(0.5 * df, 0.5, beta_value))
