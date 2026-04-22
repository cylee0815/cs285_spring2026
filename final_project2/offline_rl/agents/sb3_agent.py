"""
SB3 agent factory and portfolio evaluation callback.

Supported algorithms:
  a2c      — A2C (on-policy, synchronous actor-critic)
  ppo_sb3  — SB3 PPO (comparison baseline against the custom PPO)
  sac_sb3  — SAC (off-policy, max-entropy, continuous actions)
  td3      — TD3 (off-policy, twin delayed DDPG)
  ddpg     — DDPG (off-policy, deterministic policy gradient)
  tqc      — TQC (distributional SAC; requires `sb3-contrib`)

Algorithms intentionally NOT implemented:
  DQN   — requires Discrete action space; portfolio allocation is continuous.
  HER   — requires GoalEnv interface (desired_goal/achieved_goal); not applicable here.

Action space note:
  SB3 2.x asserts finite action space bounds for ALL continuous-action algorithms.
  Always wrap PortfolioEnv with ActionBoundedWrapper(-10, 10) before passing to any SB3 algo.
  The env clips logits to [-20, 20] before softmax, so [-10, 10] bounds don't change behavior.
"""
import numpy as np
import wandb
from typing import Optional
import gymnasium as gym

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


# SB3 2.x requires finite action space bounds for ALL continuous-action algorithms
# (assertion added in base_class.py).  Apply ActionBoundedWrapper to every algo.
BOUNDED_ALGOS = {"a2c", "ppo_sb3", "sac_sb3", "td3", "ddpg", "tqc"}


def make_sb3_model(algo_name: str, env: gym.Env, config, device: str = "auto"):
    """
    Instantiate an SB3 (or sb3-contrib) model with config-driven hyperparameters.

    Args:
        algo_name: one of {a2c, ppo_sb3, sac_sb3, td3, ddpg, tqc}
        env: training environment (use ActionBoundedWrapper for bounded algos)
        config: ml_collections.ConfigDict with algorithm-specific fields
        device: torch device string ("cpu", "cuda", or "auto")
    """
    hidden = [config.hidden_dim] * config.n_layers

    # On-policy: shared pi/vf architecture; Off-policy: separate pi/qf heads
    if algo_name in ("a2c", "ppo_sb3"):
        policy_kwargs = dict(net_arch=dict(pi=hidden, vf=hidden))
    else:
        policy_kwargs = dict(net_arch=dict(pi=hidden, qf=hidden))

    base_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.lr,
        gamma=config.gamma,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
    )

    if algo_name == "a2c":
        from stable_baselines3 import A2C
        return A2C(
            **base_kwargs,
            n_steps=config.n_steps,
            gae_lambda=config.gae_lambda,
            ent_coef=config.entropy_coef,
            vf_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
        )

    elif algo_name == "ppo_sb3":
        from stable_baselines3 import PPO
        return PPO(
            **base_kwargs,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_eps,
            ent_coef=config.entropy_coef,
            vf_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
        )

    elif algo_name == "sac_sb3":
        from stable_baselines3 import SAC
        return SAC(
            **base_kwargs,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            tau=config.polyak_tau,
            ent_coef="auto",
            train_freq=1,
            gradient_steps=1,
            learning_starts=config.learning_starts,
        )

    elif algo_name == "td3":
        from stable_baselines3 import TD3
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=config.action_noise_std * np.ones(n_actions),
        )
        return TD3(
            **base_kwargs,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            tau=config.polyak_tau,
            action_noise=action_noise,
            train_freq=1,
            gradient_steps=1,
            learning_starts=config.learning_starts,
            policy_delay=config.policy_delay,
        )

    elif algo_name == "ddpg":
        from stable_baselines3 import DDPG
        n_actions = env.action_space.shape[0]
        # OU noise is the original DDPG noise; sigma matches FinRL defaults
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=config.action_noise_std * np.ones(n_actions),
        )
        return DDPG(
            **base_kwargs,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            tau=config.polyak_tau,
            action_noise=action_noise,
            train_freq=1,
            gradient_steps=1,
            learning_starts=config.learning_starts,
        )

    elif algo_name == "tqc":
        try:
            from sb3_contrib import TQC
        except ImportError:
            raise ImportError(
                "TQC requires sb3-contrib. Add it to the project and run `uv sync`:\n"
                "  uv add sb3-contrib\n"
                "or manually edit pyproject.toml to include 'sb3-contrib>=2.0'."
            )
        return TQC(
            **base_kwargs,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            tau=config.polyak_tau,
            ent_coef="auto",
            train_freq=1,
            gradient_steps=1,
            learning_starts=config.learning_starts,
            top_quantiles_to_drop_per_net=config.top_quantiles_to_drop,
        )

    else:
        raise ValueError(
            f"Unknown SB3 algo '{algo_name}'. "
            f"Valid choices: a2c, ppo_sb3, sac_sb3, td3, ddpg, tqc.\n"
            f"Note: DQN requires Discrete actions; HER requires GoalEnv — neither fits portfolio optimization."
        )


class PortfolioEvalCallback(BaseCallback):
    """
    Callback that runs periodic evaluation on the test environment and logs
    portfolio-specific metrics (annual return, portfolio value, turnover) to WandB.

    Fires every `eval_interval` timesteps.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_interval: int,
        n_eval_episodes: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self._last_eval_step = 0

        # Try to get episode_length from env for annualization
        try:
            self._ep_len = eval_env.episode_length
        except AttributeError:
            try:
                self._ep_len = eval_env.unwrapped.episode_length
            except AttributeError:
                self._ep_len = 63  # default: ~1 quarter

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_interval:
            self._last_eval_step = self.num_timesteps
            metrics = self._run_eval()
            wandb.log({**metrics, "timesteps": self.num_timesteps})
            if self.verbose:
                ar = metrics.get("eval/annual_return", 0.0)
                pv = metrics.get("eval/portfolio_value", 1.0)
                print(f"  [eval @ {self.num_timesteps}] annual_return={ar:.2%}  portfolio_value={pv:.4f}")
        return True

    def _run_eval(self):
        episode_returns, portfolio_values, turnovers = [], [], []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_return = 0.0
            ep_turnover = 0.0
            info = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_return += float(reward)
                ep_turnover += info.get("turnover", 0.0)

            episode_returns.append(ep_return)
            portfolio_values.append(info.get("portfolio_value", 1.0))
            turnovers.append(ep_turnover)

        annual_returns = [(pv - 1.0) * (252 / self._ep_len) for pv in portfolio_values]

        return {
            "eval/episode_return": float(np.mean(episode_returns)),
            "eval/portfolio_value": float(np.mean(portfolio_values)),
            "eval/annual_return": float(np.mean(annual_returns)),
            "eval/avg_turnover": float(np.mean(turnovers)),
            "eval/std_episode_return": float(np.std(episode_returns)),
        }
