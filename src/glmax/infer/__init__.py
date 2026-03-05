from .optimize import (
    irls as irls,
)
from .solve import (
    CGSolver as CGSolver,
    CholeskySolver as CholeskySolver,
    QRSolver as QRSolver,
)
from .stderr import (
    AbstractStdErrEstimator as AbstractStdErrEstimator,
    FisherInfoError as FisherInfoError,
    HuberError as HuberError,
)


def infer(*args, **kwargs):
    """Canonical infer verb entrypoint."""
    from .inference import infer as _infer

    return _infer(*args, **kwargs)


def check(*args, **kwargs):
    """Canonical check verb placeholder until Phase 4 implementation lands."""
    del args, kwargs
    raise NotImplementedError("`check` is not implemented yet. Planned in implementation plan Phase 4.")
