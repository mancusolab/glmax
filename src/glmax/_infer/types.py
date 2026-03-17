from typing import NamedTuple

from jaxtyping import Array

from .._fit import Params


class InferenceResult(NamedTuple):
    r"""Canonical output contract for the `infer(...)` grammar verb.

    A lightweight immutable container carrying the inferential summaries
    produced by an `AbstractTest` strategy.

    **Fields:**

    - `params`: `Params` from the fitted model ($\hat\beta$, $\hat\phi$).
    - `se`: standard error vector, shape `(p,)`. May be `NaN` for strategies
      that do not compute standard errors (e.g. `ScoreTest`).
    - `stat`: test statistic vector, shape `(p,)`.
    - `p`: two-sided p-value vector, shape `(p,)`.
    """

    params: Params
    se: Array
    stat: Array
    p: Array
