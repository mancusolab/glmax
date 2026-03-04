from abc import abstractmethod

import equinox as eqx

from jax import Array
from jax.scipy.stats import norm

from ..family.dist import ExponentialFamily, Gaussian
from ..family.utils import t_cdf


class AbstractHypothesisTest(eqx.Module, strict=True):
    @abstractmethod
    def __call__(self, statistic: Array, df: int, family: ExponentialFamily) -> Array:
        pass


class WaldTest(AbstractHypothesisTest, strict=True):
    def __call__(self, statistic: Array, df: int, family: ExponentialFamily) -> Array:
        if isinstance(family, Gaussian):
            return 2 * t_cdf(-abs(statistic), df)
        return 2 * norm.sf(abs(statistic))
