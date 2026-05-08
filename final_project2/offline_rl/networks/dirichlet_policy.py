"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.networks.dirichlet_policy is deprecated; "
    "use core.networks.dirichlet_policy",
    DeprecationWarning,
    stacklevel=2,
)

from core.networks.dirichlet_policy import *  # noqa: F401,F403
