from typing import NamedTuple

from jaxtyping import Array

from .._fit import Params


class InferenceResult(NamedTuple):
    r"""Canonical output contract for the `infer(...)` grammar verb.

    A lightweight immutable container carrying the inferential summaries
    produced by a [`glmax.AbstractTest`][] strategy. The tuple stores
    $(\hat{\theta}, \operatorname{SE}, z, p)$, where $\hat{\theta}$ is the
    fitted parameter carrier, $\operatorname{SE}$ is the per-coefficient
    standard error vector, $z$ is the per-coefficient test statistic, and
    $p$ is the per-coefficient two-sided p-value.

    **Arguments:**

    - `params`: fitted [`glmax.Params`][] carrying
      $(\hat{\beta}, \hat{\phi}, \hat{a})$.
    - `se`: standard error vector, shape `(p,)`. May be `NaN` for strategies
      that do not compute standard errors (e.g. `ScoreTest`).
    - `stat`: test statistic vector, shape `(p,)`.
    - `p`: two-sided p-value vector, shape `(p,)`.
    """

    params: Params
    se: Array
    stat: Array
    p: Array
