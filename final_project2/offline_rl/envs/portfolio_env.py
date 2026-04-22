"""DEPRECATED — this module has been retired.

The canonical portfolio environment is ``env.portfolio_env.PortfolioEnv``
(at the project root). All offline-RL pipelines now construct that class
instead.

Importing from this module raises ``ImportError`` with a pointer to the
correct location so nothing silently uses the old code.
"""

raise ImportError(
    "offline_rl.envs.portfolio_env has been retired. "
    "Use 'from env.portfolio_env import PortfolioEnv' instead. "
    "See the migration notes in the commit that introduced this stub."
)
