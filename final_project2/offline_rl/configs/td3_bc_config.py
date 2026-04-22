import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.hidden_dim = 256
    config.n_layers = 2
    config.lr = 3e-4
    config.gamma = 0.99
    config.n_step = 1
    config.polyak_tau = 0.005
    config.batch_size = 256
    config.max_grad_norm = 1.0
    # TD3+BC specific
    config.td3bc_alpha = 2.5          # BC regularization weight
    config.policy_noise = 0.2         # target policy smoothing noise std
    config.noise_clip = 0.5
    config.policy_delay = 2           # actor update every N critic updates
    config.n_offline_updates = 100_000
    config.offline_buffer_size = 200_000
    return config
