"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.networks.bayesian_regime_encoder is deprecated; "
    "use core.networks.bayesian_regime_encoder",
    DeprecationWarning,
    stacklevel=2,
)

from core.networks.bayesian_regime_encoder import *  # noqa: F401,F403
