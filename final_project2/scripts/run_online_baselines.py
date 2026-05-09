"""Online-RL baselines for portfolio allocation.

Trains an online agent (PPO-Dirichlet, PPO-Dirichlet-LSTM, or SAC-Dirichlet)
against a ``PortfolioEnv`` built from the same causal-feature dataset used by
the offline IQL pipeline (``datasets/real_dirichlet.npz``). Because the
dataset already carries per-row ``forward_returns`` and ``dates``, we simply
slice it into train / val / test windows using :mod:`data.splits` — no
network required, and the online baseline evaluates on exactly the same
price path the offline model is tested on.

Usage
-----
    uv run python scripts/run_online_baselines.py \\
        --algo ppo --seed 0 \\
        --dataset datasets/real_dirichlet.npz \\
        --total_timesteps 200_000

    uv run python scripts/run_online_baselines.py --algo ppo_lstm --seed 1
    uv run python scripts/run_online_baselines.py --algo sac_dirichlet --seed 2

Outputs: ``results/online/<algo>_seed<k>/{metrics.json,weights.npy,returns.npy}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import ml_collections

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.splits import SplitConfig, compute_split_indices
from core.envs.portfolio_env import PortfolioEnv
from utils.seed import resolve_device, set_seed

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Dataset / env construction
# ---------------------------------------------------------------------------


def _load_dataset_arrays(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    states = data["states"].astype(np.float32)
    if "forward_returns" not in data.files:
        raise ValueError(
            f"Dataset {path} lacks 'forward_returns' — rebuild with "
            f"scripts/build_real_dataset.py so the online env has a return path."
        )
    fwd = data["forward_returns"].astype(np.float32)
    # build_real_dataset saves forward_returns with shape (T+1, N) vs states
    # (T, F). Trim so both align row-for-row.
    if fwd.shape[0] != states.shape[0]:
        fwd = fwd[: states.shape[0]]
    if "dates" not in data.files:
        raise ValueError(
            f"Dataset {path} lacks 'dates' — cannot compute calendar splits."
        )
    dates = data["dates"].astype(np.int64)
    return states, fwd, dates


def _slice_env(
    states: np.ndarray,
    fwd: np.ndarray,
    indices: np.ndarray,
    episode_length: int | None,
    transaction_cost: float,
) -> PortfolioEnv:
    """Build a PortfolioEnv over a (contiguous, chronological) index slice.

    ``data.splits`` guarantees each split is monotonically increasing and
    disjoint, but not necessarily contiguous in the underlying array. We
    still pass the full slice because ``PortfolioEnv`` steps by contiguous
    index — in practice train/val/test are contiguous because the dataset
    was rolled out as a single trajectory.
    """
    lo, hi = int(indices[0]), int(indices[-1]) + 1
    return PortfolioEnv(
        features=states[lo:hi],
        forward_returns=fwd[lo:hi],
        transaction_cost_lambda=transaction_cost,
        episode_length=episode_length,
    )


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _ppo_config(arch: str, reward_scale: float, hidden_dim: int) -> ml_collections.ConfigDict:
    cfg = ml_collections.ConfigDict()
    cfg.arch = arch  # "dirichlet" or "dirichlet_lstm"
    cfg.lr = 3e-4
    cfg.n_steps = 2048
    cfg.n_epochs = 10 if arch == "dirichlet" else 4
    cfg.batch_size = 64
    cfg.gamma = 0.99
    cfg.gae_lambda = 0.95
    cfg.clip_eps = 0.2
    cfg.value_coef = 0.5
    # Dirichlet entropy is on the simplex and already meaningful; a small
    # positive coefficient prevents premature alpha collapse.
    cfg.entropy_coef = 0.005
    cfg.max_grad_norm = 0.5
    cfg.hidden_dim = hidden_dim
    cfg.n_layers = 2
    cfg.lstm_layers = 1
    cfg.reward_scale = reward_scale
    return cfg


def _sac_config(hidden_dim: int) -> ml_collections.ConfigDict:
    cfg = ml_collections.ConfigDict()
    cfg.hidden_dim = hidden_dim
    cfg.regime_dim = 64
    cfg.regime_window = 20
    cfg.lr = 3e-4
    cfg.gamma = 0.99
    cfg.polyak_tau = 0.005
    cfg.batch_size = 256
    cfg.buffer_size = 100_000
    cfg.max_grad_norm = 1.0
    cfg.bayesian = False
    return cfg


# ---------------------------------------------------------------------------
# Training drivers
# ---------------------------------------------------------------------------


def _train_ppo(
    arch: str,
    train_env: PortfolioEnv,
    val_env: PortfolioEnv,
    test_env: PortfolioEnv,
    total_timesteps: int,
    eval_interval: int,
    device: torch.device,
    reward_scale: float,
    hidden_dim: int,
    log_fn,
):
    from online_rl.agents.ppo import PPOAgent

    cfg = _ppo_config(arch=arch, reward_scale=reward_scale, hidden_dim=hidden_dim)
    agent = PPOAgent(train_env, cfg, device)

    n_params = sum(p.numel() for p in agent.policy.parameters())
    log_fn(f"  policy={arch}  params={n_params:,}  reward_scale={reward_scale}")

    timesteps = 0
    next_eval = eval_interval
    while timesteps < total_timesteps:
        metrics = agent.update()
        timesteps += cfg.n_steps
        if timesteps >= next_eval:
            val_metrics = agent.evaluate(
                val_env, n_episodes=1, deterministic=True, randomize=False,
            )
            log_fn(
                f"  step={timesteps:>8d}  "
                f"train_ret={metrics.get('episode_return', 0):+.4f}  "
                f"val_sharpe={val_metrics['eval/sharpe_ratio']:+.3f}  "
                f"val_ann={val_metrics['eval/annual_return']:+.3f}  "
                f"val_mdd={val_metrics['eval/max_drawdown']:.3f}  "
                f"turnover={metrics.get('rollout/turnover', 0):.3f}  "
                f"HHI={metrics.get('rollout/hhi', 0):.3f}"
            )
            next_eval += eval_interval
    return agent


def _train_sac_dirichlet(
    train_env: PortfolioEnv,
    val_env: PortfolioEnv,
    test_env: PortfolioEnv,
    total_timesteps: int,
    eval_interval: int,
    device: torch.device,
    hidden_dim: int,
    log_fn,
):
    from online_rl.agents.sac_dirichlet import SACDirichlet

    cfg = _sac_config(hidden_dim=hidden_dim)
    agent = SACDirichlet(train_env, cfg, device)

    n_params = (
        sum(p.numel() for p in agent.actor.parameters())
        + sum(p.numel() for p in agent.critic.parameters())
    )
    log_fn(f"  policy=sac_dirichlet  params={n_params:,}")

    timesteps = 0
    next_eval = eval_interval
    last_metrics: dict = {}
    while timesteps < total_timesteps:
        m = agent.update() or {}
        if m:
            last_metrics = m
        timesteps += 1
        if timesteps >= next_eval:
            val = agent.evaluate(val_env, n_episodes=1)
            log_fn(
                f"  step={timesteps:>8d}  "
                f"val_sharpe={val.get('eval/sharpe_ratio', 0):+.3f}  "
                f"val_ann={val.get('eval/annual_return', 0):+.3f}  "
                f"val_mdd={val.get('eval/max_drawdown', 0):.3f}  "
                f"tau={last_metrics.get('temperature', 0):.3f}  "
                f"entropy={last_metrics.get('entropy', 0):+.3f}"
            )
            next_eval += eval_interval
    return agent


# ---------------------------------------------------------------------------
# Test-window backtest
# ---------------------------------------------------------------------------


def _run_test_backtest(agent, test_env: PortfolioEnv, deterministic: bool = True):
    """Single chronological sweep of the test window, recording everything."""
    obs, _ = test_env.reset(options={"randomize": False})
    device = (
        next(agent.policy.parameters()).device
        if hasattr(agent, "policy")
        else next(agent.actor.parameters()).device
    )
    weights_list: list[np.ndarray] = []
    returns_list: list[float] = []
    pv_traj: list[float] = [1.0]

    hidden = None
    # SAC-Dirichlet exposes (actor, regime_encoder) — detect and route.
    is_sac = hasattr(agent, "regime_encoder")

    if is_sac:
        gru_hidden = agent.regime_encoder.init_hidden(1, device)

    terminated = False
    while not terminated:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if is_sac:
                regime, gru_hidden = agent.regime_encoder.encode_step(obs_t, gru_hidden)
                w, _, _, _ = agent.actor(obs_t, regime, deterministic=True)
                action = w.squeeze(0).cpu().numpy()
            else:
                # PPO path. For Dirichlet policies we want the mean for eval.
                if agent.is_dirichlet and deterministic:
                    if agent.is_lstm:
                        features, new_hidden = agent.policy._forward(obs_t, hidden)
                    else:
                        features, new_hidden = agent.policy.shared(obs_t), None
                    dist = agent.policy._dist(features)
                    action = dist.mean.squeeze(0).cpu().numpy()
                    hidden = new_hidden
                else:
                    a, _, _, _, new_hidden = agent.policy.get_action_and_value(
                        obs_t, hidden=hidden,
                    )
                    action = a.squeeze(0).cpu().numpy()
                    hidden = new_hidden

        obs, _, terminated, truncated, info = test_env.step(action)
        terminated = terminated or truncated
        weights_list.append(info["executed_weights"].copy())
        returns_list.append(float(info["portfolio_return"]))
        pv_traj.append(float(info["portfolio_value"]))

    weights = np.asarray(weights_list, dtype=np.float64)
    returns = np.asarray(returns_list, dtype=np.float64)
    pv = np.asarray(pv_traj, dtype=np.float64)

    # Metrics — same conventions as evaluation.metrics.compute_all_metrics.
    running_max = np.maximum.accumulate(pv)
    dd = (running_max - pv) / np.maximum(running_max, 1e-12)
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(252))
        if returns.size > 1 and returns.std() > 1e-12 else 0.0
    )
    ann_ret = float(returns.mean() * 252)
    ann_vol = float(returns.std() * np.sqrt(252))
    turnover = (
        float(np.mean(np.abs(weights[1:] - weights[:-1]).sum(axis=1)))
        if weights.shape[0] >= 2 else 0.0
    )
    cum_ret = float(np.prod(1 + returns) - 1)

    return {
        "metrics": {
            "sharpe_ratio": sharpe,
            "annual_return": ann_ret,
            "annual_volatility": ann_vol,
            "max_drawdown": float(dd.max()),
            "turnover": turnover,
            "cumulative_return": cum_ret,
            "n_steps": int(returns.size),
        },
        "weights": weights,
        "returns": returns,
        "equity_curve": pv,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online-RL portfolio baselines.")
    p.add_argument("--algo", choices=["ppo", "ppo_lstm", "sac_dirichlet"], default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", type=str, default="datasets/real_dirichlet.npz")
    p.add_argument("--train_start", type=str, default="2008-01-01")
    p.add_argument("--train_end", type=str, default="2020-12-31")
    p.add_argument("--val_start", type=str, default="2021-01-01")
    p.add_argument("--val_end", type=str, default="2021-12-31")
    p.add_argument("--test_start", type=str, default="2022-01-01")
    p.add_argument("--test_end", type=str, default="2026-03-31")
    p.add_argument("--episode_length", type=int, default=63,
                   help="Training-window episode length in trading days.")
    p.add_argument("--transaction_cost", type=float, default=0.001)
    p.add_argument("--total_timesteps", type=int, default=200_000)
    p.add_argument("--eval_interval", type=int, default=20_000)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--reward_scale", type=float, default=100.0,
                   help="PPO reward scale. Daily returns are ~1e-2 so 100x "
                        "gives value-head targets of O(1).")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--results_dir", type=str, default="results/online")
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(resolve_device(args.device))

    # --- Dataset + splits ---------------------------------------------------
    states, fwd, dates_ns = _load_dataset_arrays(args.dataset)
    split_cfg = SplitConfig(
        train_start=args.train_start, train_end=args.train_end,
        val_start=args.val_start, val_end=args.val_end,
        test_start=args.test_start, test_end=args.test_end,
    )
    split_idx = compute_split_indices(dates_ns, split_cfg)
    print(
        f"[data] state_dim={states.shape[1]}  n_assets={fwd.shape[1]}  "
        f"train={split_idx.train.size}  val={split_idx.val.size}  "
        f"test={split_idx.test.size}"
    )

    # --- Envs ---------------------------------------------------------------
    train_env = _slice_env(
        states, fwd, split_idx.train,
        episode_length=args.episode_length,
        transaction_cost=args.transaction_cost,
    )
    # Val / test envs run as a single chronological sweep (episode_length=None
    # → T-1 steps). No random start: we want deterministic backtests.
    val_env = _slice_env(
        states, fwd, split_idx.val,
        episode_length=None, transaction_cost=args.transaction_cost,
    )
    test_env = _slice_env(
        states, fwd, split_idx.test,
        episode_length=None, transaction_cost=args.transaction_cost,
    )
    print(
        f"[envs] train.obs_dim={train_env.observation_space.shape[0]}  "
        f"val.T={val_env._n_steps}  test.T={test_env._n_steps}"
    )

    # --- Train --------------------------------------------------------------
    def _log(msg: str) -> None:
        print(msg, flush=True)

    if args.algo == "ppo":
        agent = _train_ppo(
            arch="dirichlet",
            train_env=train_env, val_env=val_env, test_env=test_env,
            total_timesteps=args.total_timesteps,
            eval_interval=args.eval_interval,
            device=device,
            reward_scale=args.reward_scale,
            hidden_dim=args.hidden_dim,
            log_fn=_log,
        )
    elif args.algo == "ppo_lstm":
        agent = _train_ppo(
            arch="dirichlet_lstm",
            train_env=train_env, val_env=val_env, test_env=test_env,
            total_timesteps=args.total_timesteps,
            eval_interval=args.eval_interval,
            device=device,
            reward_scale=args.reward_scale,
            hidden_dim=args.hidden_dim,
            log_fn=_log,
        )
    elif args.algo == "sac_dirichlet":
        agent = _train_sac_dirichlet(
            train_env=train_env, val_env=val_env, test_env=test_env,
            total_timesteps=args.total_timesteps,
            eval_interval=args.eval_interval,
            device=device,
            hidden_dim=args.hidden_dim,
            log_fn=_log,
        )
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    # --- Test backtest ------------------------------------------------------
    out = _run_test_backtest(agent, test_env, deterministic=True)
    m = out["metrics"]
    print("=" * 60)
    print(f"[TEST] algo={args.algo}  seed={args.seed}")
    for k in ["sharpe_ratio", "annual_return", "annual_volatility",
              "max_drawdown", "turnover", "cumulative_return", "n_steps"]:
        print(f"  {k:20s}= {m[k]:+.4f}" if isinstance(m[k], float) else f"  {k:20s}= {m[k]}")
    print("=" * 60)

    # --- Persist ------------------------------------------------------------
    run_name = args.run_name or f"{args.algo}_seed{args.seed}"
    out_dir = Path(args.results_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "algo": args.algo, "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "episode_length": args.episode_length,
            "transaction_cost": args.transaction_cost,
            "reward_scale": args.reward_scale,
            "split": split_cfg.__dict__,
            "test": m,
        }, f, indent=2)
    np.save(out_dir / "weights.npy", out["weights"])
    np.save(out_dir / "returns.npy", out["returns"])
    np.save(out_dir / "equity_curve.npy", out["equity_curve"])
    np.save(out_dir / "dates.npy", dates_ns[split_idx.test][: out["returns"].size])
    print(f"[save] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
