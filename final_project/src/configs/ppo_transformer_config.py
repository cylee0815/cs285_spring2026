import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.arch = "transformer"
    config.lr = 3e-4
    config.n_steps = 2048
    config.n_epochs = 4
    config.batch_size = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_eps = 0.2
    config.value_coef = 0.5
    config.entropy_coef = 0.01
    config.max_grad_norm = 0.5
    config.context_len = 20        # Transformer context window (trading days)
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 2
    return config
