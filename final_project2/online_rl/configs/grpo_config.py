"""Default config for Continuous GRPO (online_rl.agents.grpo.GRPOTrainer).

Mirrors the ``ml_collections.ConfigDict`` factory pattern used by
``ppo_config.py`` so the registry in ``offline_rl.configs.__init__`` can
load both the same way.
"""
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    # GRPO-specific
    config.group_size = 16            # samples per state (G)
    config.advantage_norm = "mean_std"  # one of: raw, mean_only, mean_std, rank
    config.beta_kl = 0.01             # KL-to-ref-policy coefficient
    config.clip_eps = 0.2             # PPO-style importance-ratio clip
    config.epochs_per_batch = 4       # passes over each collected batch
    config.minibatch_size = 256       # in states (not state-sample pairs)
    config.lr = 3e-4
    config.grad_clip = 1.0
    config.entropy_coef = 0.0

    # Training schedule
    config.total_env_steps = 200_000
    config.states_per_collect = 2048  # smoke runs may want --states_per_collect 512
    config.log_every = 1

    # Actor
    config.actor_type = "DirichletMLPPolicy"  # alt: "DirichletLSTMPolicy"
    config.hidden_dim = 256
    config.n_layers = 2
    return config
