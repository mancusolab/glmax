"""
    Dummy conftest.py for glmax.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
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
