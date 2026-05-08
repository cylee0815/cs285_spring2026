"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.configs.bayesian_o2o_config is deprecated; "
    "use hybrid_rl.configs.bayesian_o2o_config",
    DeprecationWarning,
    stacklevel=2,
)

from hybrid_rl.configs.bayesian_o2o_config import *  # noqa: F401,F403
