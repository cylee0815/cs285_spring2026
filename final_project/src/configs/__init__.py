CONFIG_MAP = {
    # Online PPO variants (architecture comparison)
    "ppo": "src.configs.ppo_config",
    "ppo_lstm": "src.configs.ppo_lstm_config",
    "ppo_transformer": "src.configs.ppo_transformer_config",
    # Novel domain-specific algorithms
    "sac_dirichlet": "src.configs.sac_dirichlet_config",    # Online SAC-Dirichlet baseline
    "cql_geodesic": "src.configs.cql_geodesic_config",      # Offline Geodesic-CQL
    "o2o": "src.configs.o2o_config",                         # Full O2O pipeline
}
