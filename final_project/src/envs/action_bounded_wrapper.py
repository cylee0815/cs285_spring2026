"""
ActionBoundedWrapper: clips the action space to a bounded Box for SB3 compatibility.

SAC, TD3, and DDPG in SB3 require a *bounded* action space because they compute:
    action_scale = (high - low) / 2    →  inf if high=+inf
inside their actor heads. Setting bounds to [-10, 10] is safe here because
PortfolioEnv clips logits to [-20, 20] before softmax anyway, so no behavior changes.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ActionBoundedWrapper(gym.Wrapper):
    """Replaces the action space with Box(-bound, +bound) while passing actions through."""

    def __init__(self, env: gym.Env, bound: float = 10.0):
        super().__init__(env)
        n = env.action_space.shape[0]
        self.action_space = spaces.Box(
            low=-bound, high=bound, shape=(n,), dtype=np.float32
        )

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
