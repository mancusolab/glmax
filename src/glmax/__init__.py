# pattern: Imperative Shell

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
    GLMState as GLMState,
)
from .infer import (
    CGSolver as CGSolver,
    CholeskySolver as CholeskySolver,
    FisherInfoError as FisherInfoError,
    HuberError as HuberError,
    irls as irls,
    QRSolver as QRSolver,
)


jax.config.update("jax_enable_x64", True)  # noqa: E402

__version__ = version("glmax")
