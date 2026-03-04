from typing import NamedTuple

from jax import Array


class GLMState(NamedTuple):
    beta: Array
    se: Array
    z: Array
    p: Array
    eta: Array
    mu: Array
    glm_wt: Array
    num_iters: Array
    converged: Array
    infor_inv: Array
    resid: Array
    alpha: Array
