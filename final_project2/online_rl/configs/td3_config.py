import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.arch = "mlp"
    config.hidden_dim = 256
    config.n_layers = 2
    config.lr = 1e-3
    config.gamma = 0.99
    config.polyak_tau = 0.005
    config.batch_size = 256
    config.buffer_size = 100_000
    config.learning_starts = 1_000
    config.action_noise_std = 0.1   # Gaussian noise std added during training
    config.policy_delay = 2         # actor update every 2 critic updates
    return config
