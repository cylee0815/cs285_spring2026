import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    # Architecture (matches GeodesicCQL for fair ablation)
    config.hidden_dim = 256
    config.regime_dim = 64
    config.regime_window = 20
    config.lr = 3e-4
    config.gamma = 0.99
    config.n_step = 1
    config.polyak_tau = 0.005
    config.batch_size = 256
    config.max_grad_norm = 1.0
    # CQL penalty
    config.cql_alpha = 1.0         # standard CQL penalty weight
    config.cql_n_samples = 10      # samples for policy Q estimation
    config.n_offline_updates = 100_000
    config.offline_buffer_size = 200_000
    return config
