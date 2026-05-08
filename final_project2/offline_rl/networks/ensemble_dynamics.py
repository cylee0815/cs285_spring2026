"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.networks.ensemble_dynamics is deprecated; "
    "use core.networks.ensemble_dynamics",
    DeprecationWarning,
    stacklevel=2,
)

from core.networks.ensemble_dynamics import *  # noqa: F401,F403
