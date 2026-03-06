# pattern: Imperative Shell

"""Pytest-only test bootstrap for this worktree.

Use ``tools/worktree-python`` for non-pytest commands that must import ``glmax``
from this worktree's ``src/`` tree.
"""
import sys

from pathlib import Path

import pytest

import equinox.internal as eqxi
import jax


PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def getkey():
    return eqxi.GetKey()
