"""Temporary import shim. DELETE IN PHASE 5."""

import warnings

warnings.warn(
    "offline_rl.agents.replay_buffer is deprecated; use core.buffers.replay_buffer",
    DeprecationWarning,
    stacklevel=2,
)

from core.buffers.replay_buffer import *  # noqa: F401,F403
