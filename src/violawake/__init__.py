"""Compatibility shim: ``import violawake`` works as an alias for ``violawake_sdk``."""

from violawake_sdk import *  # noqa: F401,F403
from violawake_sdk import __version__  # noqa: F401
