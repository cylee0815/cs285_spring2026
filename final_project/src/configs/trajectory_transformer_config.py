import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.context_len = 20
    config.d_model = 128
    config.n_heads = 4
    config.n_layers = 4
    config.dropout = 0.1
    config.lr = 1e-4
    config.batch_size = 64
    config.max_grad_norm = 0.25
    config.n_bins = 32           # discretization bins per dimension
    config.beam_width = 32       # beam search width at inference
    config.n_offline_updates = 100_000
    config.offline_buffer_size = 200_000
    return config
