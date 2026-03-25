import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.arch = "mlp"
    config.lr = 3e-4
    config.n_steps = 2048          # steps per rollout collection
    config.n_epochs = 10           # PPO update epochs per rollout
    config.batch_size = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_eps = 0.2
    config.value_coef = 0.5
    config.entropy_coef = 0.01
    config.max_grad_norm = 0.5
    config.hidden_dim = 256
    config.n_layers = 2
    return config
