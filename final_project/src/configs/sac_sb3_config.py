import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.arch = "mlp"
    config.hidden_dim = 256
    config.n_layers = 2
    config.lr = 3e-4
    config.gamma = 0.99
    config.polyak_tau = 0.005
    config.batch_size = 256
    config.buffer_size = 100_000
    config.learning_starts = 1_000   # steps before first gradient update
    return config
