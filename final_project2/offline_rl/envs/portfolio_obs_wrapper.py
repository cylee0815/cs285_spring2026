"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.envs.portfolio_obs_wrapper is deprecated; "
    "use core.envs.portfolio_obs_wrapper",
    DeprecationWarning,
    stacklevel=2,
)

from core.envs.portfolio_obs_wrapper import *  # noqa: F401,F403
