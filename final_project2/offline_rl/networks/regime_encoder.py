"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.networks.regime_encoder is deprecated; "
    "use core.networks.regime_encoder",
    DeprecationWarning,
    stacklevel=2,
)

from core.networks.regime_encoder import *  # noqa: F401,F403
