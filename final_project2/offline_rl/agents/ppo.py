"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.agents.ppo is deprecated; use online_rl.agents.ppo",
    DeprecationWarning,
    stacklevel=2,
)

from online_rl.agents.ppo import *  # noqa: F401,F403
