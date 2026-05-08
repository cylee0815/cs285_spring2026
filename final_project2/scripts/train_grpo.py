"""Train a Continuous GRPO actor on the portfolio-allocation task.

Mirrors ``scripts/run_online_baselines.py`` for everything that affects fair
comparison with the PPO baseline: dataset loader, calendar splits, env
construction, seeding, and result-directory layout. The only differences are
algorithm-specific (no critic, group-relative advantages, KL-to-ref anchor —
see ``online_rl/agents/grpo.py``).

Usage
-----
    uv run python scripts/train_grpo.py \\
        --total_env_steps 200_000 --group_size 16 --seed 0

Smoke train (catches integration / hyperparameter bugs)::

    uv run python scripts/train_grpo.py \\
        --total_env_steps 5000 --group_size 4 --seed 42 \\
        --states_per_collect 512 --device cpu

Outputs: ``results/online/grpo_seed<k>/{metrics.json,checkpoint.pt,
weights.npy,returns.npy,equity_curve.npy,dates.npy}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.splits import SplitConfig, compute_split_indices
from core.envs.portfolio_env import PortfolioEnv
from core.networks.policies import DirichletMLPPolicy
from online_rl.agents.grpo import GRPOConfig, GRPOTrainer
from online_rl.configs.grpo_config import get_config as get_grpo_config
from utils.seed import resolve_device, set_seed

__all__ = ["main"]


# ---------------------------------------------------------------------------
# Dataset / env construction (must match run_online_baselines.py)
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
    lo, hi = int(indices[0]), int(indices[-1]) + 1
    return PortfolioEnv(
        features=states[lo:hi],
        forward_returns=fwd[lo:hi],
        transaction_cost_lambda=transaction_cost,
        episode_length=episode_length,
        # GRPO requires include_prev_weights=False (Stage A exogeneity guard).
        include_prev_weights=False,
    )


# ---------------------------------------------------------------------------
# Test-window backtest (Dirichlet mean as the deterministic action)
# ---------------------------------------------------------------------------


def _run_test_backtest(
    actor: DirichletMLPPolicy,
    test_env: PortfolioEnv,
    device: torch.device,
) -> dict:
    """Single chronological sweep using the Dirichlet mean at each step."""
    obs, _ = test_env.reset(options={"randomize": False})
    actor.eval()
    weights_list: list[np.ndarray] = []
    returns_list: list[float] = []
    pv_traj: list[float] = [1.0]

    terminated = False
    while not terminated:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            dist = actor.dist(obs_t)
            action = dist.mean.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, info = test_env.step(action)
        terminated = terminated or truncated
        weights_list.append(info["executed_weights"].copy())
        returns_list.append(float(info["portfolio_return"]))
        pv_traj.append(float(info["portfolio_value"]))

    weights = np.asarray(weights_list, dtype=np.float64)
    returns = np.asarray(returns_list, dtype=np.float64)
    pv = np.asarray(pv_traj, dtype=np.float64)

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
    cfg_defaults = get_grpo_config()
    p = argparse.ArgumentParser(description="Train Continuous GRPO for portfolio allocation.")
    # Data / splits — match run_online_baselines.py defaults exactly.
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
    # Training budget
    p.add_argument("--total_env_steps", type=int, default=cfg_defaults.total_env_steps)
    p.add_argument("--states_per_collect", type=int, default=cfg_defaults.states_per_collect,
                   help="Env steps per collect/update iteration. 512 reasonable for smoke runs.")
    p.add_argument("--log_every", type=int, default=cfg_defaults.log_every,
                   help="Print one log line every N iterations.")
    # GRPO hyperparameters (overrides over the config file)
    p.add_argument("--group_size", type=int, default=cfg_defaults.group_size)
    p.add_argument("--advantage_norm", type=str, default=cfg_defaults.advantage_norm,
                   choices=["raw", "mean_only", "mean_std", "rank"])
    p.add_argument("--beta_kl", type=float, default=cfg_defaults.beta_kl)
    p.add_argument("--clip_eps", type=float, default=cfg_defaults.clip_eps)
    p.add_argument("--epochs_per_batch", type=int, default=cfg_defaults.epochs_per_batch)
    p.add_argument("--minibatch_size", type=int, default=cfg_defaults.minibatch_size)
    p.add_argument("--lr", type=float, default=cfg_defaults.lr)
    p.add_argument("--grad_clip", type=float, default=cfg_defaults.grad_clip)
    p.add_argument("--entropy_coef", type=float, default=cfg_defaults.entropy_coef)
    # Actor architecture
    p.add_argument("--hidden_dim", type=int, default=cfg_defaults.hidden_dim)
    p.add_argument("--n_layers", type=int, default=cfg_defaults.n_layers)
    # I/O
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--output_dir", type=str, default="results/online")
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
    val_env = _slice_env(
        states, fwd, split_idx.val,
        episode_length=None, transaction_cost=args.transaction_cost,
    )
    test_env = _slice_env(
        states, fwd, split_idx.test,
        episode_length=None, transaction_cost=args.transaction_cost,
    )
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    print(
        f"[envs] train.obs_dim={obs_dim}  n_assets={action_dim}  "
        f"val.T={val_env._n_steps}  test.T={test_env._n_steps}"
    )

    # --- Actor + trainer ----------------------------------------------------
    actor = DirichletMLPPolicy(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    )
    cfg = GRPOConfig(
        group_size=args.group_size,
        advantage_norm=args.advantage_norm,
        beta_kl=args.beta_kl,
        clip_eps=args.clip_eps,
        epochs_per_batch=args.epochs_per_batch,
        minibatch_size=args.minibatch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        entropy_coef=args.entropy_coef,
    )
    trainer = GRPOTrainer(actor, train_env, cfg, device=device, seed=args.seed)
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"[actor] DirichletMLPPolicy  hidden_dim={args.hidden_dim}  params={n_params:,}")
    print(
        f"[grpo]  G={cfg.group_size}  norm={cfg.advantage_norm}  "
        f"beta_kl={cfg.beta_kl}  clip_eps={cfg.clip_eps}  lr={cfg.lr}  "
        f"epochs={cfg.epochs_per_batch}  mb={cfg.minibatch_size}"
    )

    # --- Train loop ---------------------------------------------------------
    env_steps_done = 0
    iteration = 0
    while env_steps_done < args.total_env_steps:
        n = min(args.states_per_collect, args.total_env_steps - env_steps_done)
        batch = trainer.collect(num_states=n)
        metrics = trainer.update(batch)
        env_steps_done += n
        iteration += 1
        if iteration % args.log_every == 0:
            print(
                f"iter {iteration:4d} | env_steps {env_steps_done:6d} | "
                f"reward_mean {metrics['reward_mean']:+.4f} | "
                f"kl_to_ref {metrics['kl_to_ref']:.4f} | "
                f"clipfrac {metrics['clipfrac']:.3f} | "
                f"adv_std {metrics['adv_std']:.3f} | "
                f"ratio_hard_clipped {metrics['ratio_hard_clipped']:.0f}",
                flush=True,
            )

    # --- Test backtest ------------------------------------------------------
    out = _run_test_backtest(actor, test_env, device)
    m = out["metrics"]
    print("=" * 60)
    print(f"[TEST] algo=grpo  seed={args.seed}")
    for k in ["sharpe_ratio", "annual_return", "annual_volatility",
              "max_drawdown", "turnover", "cumulative_return", "n_steps"]:
        print(f"  {k:20s}= {m[k]:+.4f}" if isinstance(m[k], float) else f"  {k:20s}= {m[k]}")
    print("=" * 60)

    # --- Persist ------------------------------------------------------------
    run_name = args.run_name or f"grpo_seed{args.seed}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "actor": actor.state_dict(),
            "cfg": vars(cfg),
            "metadata": {
                "seed": args.seed,
                "env_steps_completed": env_steps_done,
                "iterations": iteration,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
            },
        },
        out_dir / "checkpoint.pt",
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "algo": "grpo", "seed": args.seed,
            "total_env_steps": args.total_env_steps,
            "states_per_collect": args.states_per_collect,
            "iterations": iteration,
            "episode_length": args.episode_length,
            "transaction_cost": args.transaction_cost,
            "grpo_cfg": vars(cfg),
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
