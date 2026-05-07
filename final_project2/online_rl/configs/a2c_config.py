import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.arch = "mlp"
    config.hidden_dim = 256
    config.n_layers = 2
    config.lr = 7e-4
    config.n_steps = 5            # rollout length per update (short for on-policy speed)
    config.gamma = 0.99
    config.gae_lambda = 1.0       # MC returns by default (no GAE decay)
    config.entropy_coef = 0.01
    config.value_coef = 0.5
    config.max_grad_norm = 0.5
    return config
