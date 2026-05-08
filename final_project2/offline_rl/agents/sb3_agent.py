"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.agents.sb3_agent is deprecated; use online_rl.agents.sb3_agent",
    DeprecationWarning,
    stacklevel=2,
)

from online_rl.agents.sb3_agent import *  # noqa: F401,F403
