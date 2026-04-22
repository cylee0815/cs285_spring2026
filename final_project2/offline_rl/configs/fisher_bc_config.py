import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.hidden_dim = 256
    config.n_layers = 2
    config.lr = 3e-4
    config.batch_size = 256
    config.max_grad_norm = 1.0
    config.n_offline_updates = 100_000
    config.offline_buffer_size = 200_000
    config.gamma = 0.99
    config.n_step = 1
    return config
