"""Temporary import shim. DELETE IN PHASE 5.

Re-exports from core.envs.portfolio_env. Emits DeprecationWarning so callers
still using ``from env.portfolio_env import ...`` get a one-time nudge.
"""

import warnings

warnings.warn(
    "env.portfolio_env is deprecated; use core.envs.portfolio_env",
    DeprecationWarning,
    stacklevel=2,
)

from core.envs.portfolio_env import *  # noqa: F401,F403
