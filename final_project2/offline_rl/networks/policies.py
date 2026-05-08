"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.networks.policies is deprecated; use core.networks.policies",
    DeprecationWarning,
    stacklevel=2,
)

from core.networks.policies import *  # noqa: F401,F403
