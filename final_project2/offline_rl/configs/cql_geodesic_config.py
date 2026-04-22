import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    # Architecture
    config.hidden_dim = 256
    config.regime_dim = 64
    config.regime_window = 20
    # SAC base hyperparams
    config.lr = 3e-4
    config.gamma = 0.99
    config.polyak_tau = 0.005
    config.batch_size = 256
    config.max_grad_norm = 1.0
    # Geodesic-CQL penalty
    config.cql_alpha = 5.0    # CQL penalty coefficient β
    return config
