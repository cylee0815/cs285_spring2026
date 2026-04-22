import ml_collections
from offline_rl.configs.mbpo_config import get_config as get_mbpo_config

def get_config():
    config = get_mbpo_config()
    config.mopo_lambda = 1.0  # uncertainty penalty coefficient
    return config
