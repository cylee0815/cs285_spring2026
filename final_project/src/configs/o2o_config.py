import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    # Architecture (shared between offline and online)
    config.hidden_dim = 256
    config.regime_dim = 64
    config.regime_window = 20
    # Optimization
    config.lr = 3e-4
    config.gamma = 0.99
    config.polyak_tau = 0.005
    config.batch_size = 256
    config.max_grad_norm = 1.0
    # Offline pre-training
    config.offline_buffer_size = 200_000
    config.n_offline_updates = 100_000
    # Online fine-tuning
    config.buffer_size = 100_000
    config.n_online_steps = 200_000
    # Geodesic-CQL penalty
    config.cql_alpha = 5.0
    # Regime KL-based conservatism annealing
    config.regime_kl_scale = 0.5   # λ in sigmoid(λ * KL)
    return config
