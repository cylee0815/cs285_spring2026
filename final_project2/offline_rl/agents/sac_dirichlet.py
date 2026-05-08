"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.agents.sac_dirichlet is deprecated; "
    "use online_rl.agents.sac_dirichlet",
    DeprecationWarning,
    stacklevel=2,
)

from online_rl.agents.sac_dirichlet import *  # noqa: F401,F403
