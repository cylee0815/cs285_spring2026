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
    # IQL-specific
    config.iql_tau = 0.7          # expectile (0.5=mean, 0.9=near-max)
    config.iql_beta = 3.0         # advantage temperature for policy extraction
    config.iql_max_weight = 100.0 # clip advantage weights
    config.n_offline_updates = 100_000
    config.offline_buffer_size = 200_000
    return config
