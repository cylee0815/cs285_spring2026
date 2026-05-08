"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.envs.macro_features is deprecated; use core.envs.macro_features",
    DeprecationWarning,
    stacklevel=2,
)

from core.envs.macro_features import *  # noqa: F401,F403
