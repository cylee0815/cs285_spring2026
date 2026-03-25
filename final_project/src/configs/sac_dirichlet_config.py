import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    # Architecture
    config.hidden_dim = 256
    config.regime_dim = 64
    config.regime_window = 20
    # SAC hyperparams
    config.lr = 3e-4
    config.gamma = 0.99
    config.polyak_tau = 0.005
    config.batch_size = 256
    config.buffer_size = 100_000
    config.max_grad_norm = 1.0
    # Dirichlet-specific: target entropy = log(n_assets) set at runtime
    return config
