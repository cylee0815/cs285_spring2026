"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.envs.sentiment_features is deprecated; "
    "use core.envs.sentiment_features",
    DeprecationWarning,
    stacklevel=2,
)

from core.envs.sentiment_features import *  # noqa: F401,F403
