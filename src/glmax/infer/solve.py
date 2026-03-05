# pattern: Imperative Shell

import warnings

from .contracts import (
    AbstractLinearSolver as AbstractLinearSolver,
    SolverState as SolverState,
)
from .solvers import (
    CGSolver as CGSolver,
    CholeskySolver as CholeskySolver,
    QRSolver as QRSolver,
)


warnings.warn(
    "`glmax.infer.solve` is deprecated; import from `glmax.infer.contracts` and `glmax.infer.solvers` instead.",
    DeprecationWarning,
    stacklevel=2,
)
