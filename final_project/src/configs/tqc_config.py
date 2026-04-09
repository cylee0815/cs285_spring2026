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
    config.learning_starts = 1_000
    # TQC-specific: number of top quantiles to drop from each critic network
    # per net (reduces overestimation bias); paper recommends 2 for most tasks
    config.top_quantiles_to_drop = 2
    return config
