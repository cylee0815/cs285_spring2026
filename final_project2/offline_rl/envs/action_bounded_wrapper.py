"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.envs.action_bounded_wrapper is deprecated; "
    "use core.envs.action_bounded_wrapper",
    DeprecationWarning,
    stacklevel=2,
)

from core.envs.action_bounded_wrapper import *  # noqa: F401,F403
