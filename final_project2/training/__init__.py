"""Training loop and experiment orchestration.

Populated in **Milestone 5**. Exposes the ``train_iql`` entrypoint which
loads a config, builds the offline replay buffer, instantiates the agent,
and runs the update loop with periodic validation backtests.
"""
