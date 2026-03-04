from typing import NamedTuple

from jax import Array


class IRLSState(NamedTuple):
    beta: Array
    num_iters: int
    converged: Array
    alpha: Array
