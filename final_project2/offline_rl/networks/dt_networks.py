"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.networks.dt_networks is deprecated; use core.networks.dt_networks",
    DeprecationWarning,
    stacklevel=2,
)

from core.networks.dt_networks import *  # noqa: F401,F403
