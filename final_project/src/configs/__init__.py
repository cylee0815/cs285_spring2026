CONFIG_MAP = {
    # Online PPO variants (architecture comparison)
    "ppo": "src.configs.ppo_config",
    "ppo_lstm": "src.configs.ppo_lstm_config",
    "ppo_transformer": "src.configs.ppo_transformer_config",
    # Novel domain-specific algorithms
    "sac_dirichlet": "src.configs.sac_dirichlet_config",    # Online SAC-Dirichlet baseline
    "cql_geodesic": "src.configs.cql_geodesic_config",      # Offline Geodesic-CQL
    "o2o": "src.configs.o2o_config",                         # Full O2O pipeline
    "bayesian_o2o": "src.configs.bayesian_o2o_config",       # Bayesian regime O2O pipeline
    # SB3 baselines (run via run_sb3.py)
    "a2c": "src.configs.a2c_config",
    "ppo_sb3": "src.configs.ppo_sb3_config",
    "sac_sb3": "src.configs.sac_sb3_config",
    "td3": "src.configs.td3_config",
    "ddpg": "src.configs.ddpg_config",
    "tqc": "src.configs.tqc_config",
    # Offline RL baselines (run via run_offline.py)
    "bc": "src.configs.bc_config",
    "fisher_bc": "src.configs.fisher_bc_config",
    "td3_bc": "src.configs.td3_bc_config",
    "awac": "src.configs.awac_config",
    "cql_vanilla": "src.configs.cql_vanilla_config",
    "iql": "src.configs.iql_config",
    "edac": "src.configs.edac_config",
    "bcq": "src.configs.bcq_config",
    # Model-based offline RL (run via run_offline.py)
    "mbpo": "src.configs.mbpo_config",
    "mopo": "src.configs.mopo_config",
    # Sequence modeling (run via run_offline.py)
    "decision_transformer": "src.configs.decision_transformer_config",
    "trajectory_transformer": "src.configs.trajectory_transformer_config",
}
