from importlib.metadata import version  # pragma: no cover

from .glm import (
    GLM as GLM,
    GLMState as GLMState,
)


__version__ = version("glmax")
