"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.configs.tqc_config is deprecated; use online_rl.configs.tqc_config",
    DeprecationWarning,
    stacklevel=2,
)

from online_rl.configs.tqc_config import *  # noqa: F401,F403
