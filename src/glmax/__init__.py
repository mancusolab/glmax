from importlib.metadata import version  # pragma: no cover

import jax

from .family import (
    Binomial as Binomial,
    ExponentialFamily as ExponentialFamily,
    Gaussian as Gaussian,
    Identity as Identity,
    Log as Log,
    Logit as Logit,
    NBlink as NBlink,
    NegativeBinomial as NegativeBinomial,
    Poisson as Poisson,
    Power as Power,
)
from .fit import (
    fit as fit,
)
from .glm import (
    GLM as GLM,
)
from .infer import (
    AbstractFitter as AbstractFitter,
    AbstractHypothesisTest as AbstractHypothesisTest,
    CGSolver as CGSolver,
    CholeskySolver as CholeskySolver,
    DefaultFitter as DefaultFitter,
    FisherInfoError as FisherInfoError,
    GLMState as GLMState,
    HuberError as HuberError,
    irls as irls,
    IRLSState as IRLSState,
    QRSolver as QRSolver,
    WaldTest as WaldTest,
)


jax.config.update("jax_enable_x64", True)  # noqa: E402

__version__ = version("glmax")
