"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.envs.finrl_wrapper is deprecated; use core.envs.finrl_wrapper",
    DeprecationWarning,
    stacklevel=2,
)

from core.envs.finrl_wrapper import *  # noqa: F401,F403
