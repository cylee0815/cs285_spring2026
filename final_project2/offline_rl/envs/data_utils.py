"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.envs.data_utils is deprecated; use core.envs.data_utils",
    DeprecationWarning,
    stacklevel=2,
)

from core.envs.data_utils import *  # noqa: F401,F403
