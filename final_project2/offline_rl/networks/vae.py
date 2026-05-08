"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.networks.vae is deprecated; use core.networks.vae",
    DeprecationWarning,
    stacklevel=2,
)

from core.networks.vae import *  # noqa: F401,F403
