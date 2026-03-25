import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.arch = "lstm"
    config.lr = 3e-4
    config.n_steps = 2048
    config.n_epochs = 4            # Fewer epochs for recurrent (avoid stale hidden states)
    config.batch_size = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_eps = 0.2
    config.value_coef = 0.5
    config.entropy_coef = 0.01
    config.max_grad_norm = 0.5
    config.hidden_dim = 256
    config.lstm_layers = 1
    return config
