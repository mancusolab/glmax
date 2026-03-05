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


def infer(*_args, **_kwargs):
    """Canonical infer verb placeholder until Phase 4 implementation lands."""
    raise NotImplementedError("`infer` is not implemented yet. Planned in implementation plan Phase 4.")


def check(*_args, **_kwargs):
    """Canonical check verb placeholder until Phase 4 implementation lands."""
    raise NotImplementedError("`check` is not implemented yet. Planned in implementation plan Phase 4.")
