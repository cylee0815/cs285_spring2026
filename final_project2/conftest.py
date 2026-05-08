"""Pytest root conftest.

The repository uses a flat-layout module structure (``data/``, ``env/``,
``features/``, ``models/``, ``algorithms/``, ``training/``, ``evaluation/``,
``utils/``) as prescribed by ``CLAUDE.md``. These modules are **not** installed
as a Python package — ``uv`` manages the virtual environment, but the project
code itself lives on ``sys.path`` directly from the project root.

Placing ``conftest.py`` at the project root has two effects:

1. pytest automatically inserts the directory containing this file onto
   ``sys.path``, so tests can do ``from core.envs.portfolio_env import PortfolioEnv``
   without any editable-install dance.
2. It marks the project root as the pytest ``rootdir`` for consistent test
   discovery regardless of the cwd from which tests are invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable. pytest normally handles this, but
# being explicit protects against edge cases (e.g. running ``python -m pytest``
# from a subdirectory or under tools that mangle sys.path).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
