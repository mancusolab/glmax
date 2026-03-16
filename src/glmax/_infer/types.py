from typing import NamedTuple

from jaxtyping import Array

from .._fit import Params


class InferenceResult(NamedTuple):
    """Canonical _infer verb output contract."""

    params: Params
    se: Array
    stat: Array
    p: Array
