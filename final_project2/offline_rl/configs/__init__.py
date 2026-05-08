"""Cross-package config registry. Despite living under offline_rl/ for historical reasons, this registry spans offline_rl, online_rl, and hybrid_rl."""

CONFIG_MAP = {
    # Online PPO variants (architecture comparison)
    "ppo": "online_rl.configs.ppo_config",
    "ppo_lstm": "online_rl.configs.ppo_lstm_config",
    "ppo_transformer": "online_rl.configs.ppo_transformer_config",
    # Novel domain-specific algorithms
    "sac_dirichlet": "online_rl.configs.sac_dirichlet_config",     # Online SAC-Dirichlet baseline
    "grpo": "online_rl.configs.grpo_config",                       # Continuous GRPO (critic-free)
    "cql_geodesic": "offline_rl.configs.cql_geodesic_config",      # Offline Geodesic-CQL
    "o2o": "hybrid_rl.configs.o2o_config",                          # Full O2O pipeline
    "bayesian_o2o": "hybrid_rl.configs.bayesian_o2o_config",        # Bayesian regime O2O pipeline
    # SB3 baselines (run via run_sb3.py)
    "a2c": "online_rl.configs.a2c_config",
    "ppo_sb3": "online_rl.configs.ppo_sb3_config",
    "sac_sb3": "online_rl.configs.sac_sb3_config",
    "td3": "online_rl.configs.td3_config",
    "ddpg": "online_rl.configs.ddpg_config",
    "tqc": "online_rl.configs.tqc_config",
    # Offline RL baselines (run via run_offline.py)
    "bc": "offline_rl.configs.bc_config",
    "fisher_bc": "offline_rl.configs.fisher_bc_config",
    "td3_bc": "offline_rl.configs.td3_bc_config",
    "awac": "offline_rl.configs.awac_config",
    "cql_vanilla": "offline_rl.configs.cql_vanilla_config",
    "iql": "offline_rl.configs.iql_config",
    "edac": "offline_rl.configs.edac_config",
    "bcq": "offline_rl.configs.bcq_config",
    # Model-based offline RL (run via run_offline.py)
    "mbpo": "offline_rl.configs.mbpo_config",
    "mopo": "offline_rl.configs.mopo_config",
    # Sequence modeling (run via run_offline.py)
    "decision_transformer": "offline_rl.configs.decision_transformer_config",
    "trajectory_transformer": "offline_rl.configs.trajectory_transformer_config",
}
