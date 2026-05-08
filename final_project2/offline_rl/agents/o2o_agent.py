"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.agents.o2o_agent is deprecated; use hybrid_rl.agents.o2o_agent",
    DeprecationWarning,
    stacklevel=2,
)

from hybrid_rl.agents.o2o_agent import *  # noqa: F401,F403
