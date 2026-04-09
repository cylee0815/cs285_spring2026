import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    # Policy / critic
    config.hidden_dim = 256
    config.n_layers = 2
    config.lr = 3e-4
    config.gamma = 0.99
    config.polyak_tau = 0.005
    config.batch_size = 256
    config.max_grad_norm = 1.0
    # Dynamics model
    config.n_ensemble = 7
    config.n_elites = 5
    config.dynamics_lr = 1e-3
    config.dynamics_hidden_dim = 400
    config.dynamics_n_layers = 4
    config.max_epochs_since_update = 5
    # Rollout
    config.rollout_length = 1
    config.rollout_batch_size = 400
    config.model_train_freq = 250
    config.real_ratio = 0.05
    config.model_buffer_size = 400_000
    # SAC inner agent
    config.regime_dim = 64
    config.regime_window = 20
    config.buffer_size = 100_000
    config.n_offline_updates = 100_000
    config.offline_buffer_size = 200_000
    return config
