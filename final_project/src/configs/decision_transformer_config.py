import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.context_len = 20
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 3
    config.dropout = 0.1
    config.lr = 1e-4
    config.batch_size = 64
    config.max_grad_norm = 0.25
    config.rtg_scale = 1.0
    config.max_rtg = 10.0
    config.n_offline_updates = 100_000
    config.offline_buffer_size = 200_000
    return config
