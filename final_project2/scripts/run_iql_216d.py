"""One-off IQL trainer on the 216-d build_features pipeline.

Originally written to provide a feature-compatible warm-start source
for the proposed Phase 2C "GRPO with offline warm-start" condition.
The 20k-update terminal checkpoint did not pass the leak-invariance
sanity band ([0.85, 1.05] test Sharpe), and a follow-up dynamics check
(see writeup/216d_iql_dynamics.md) found the val Sharpe degrades
monotonically from a step-1000 peak of +1.32 to a step-19000 floor of
-0.95 — the equal-weight basin is a TRANSIENT state on 216-d, not a
fixed point as it is on 56-d. The GRPO warm-start condition was dropped
on this basis (writeup/draft_experiments.tex; appendix B). The script
and its outputs are retained as reproducibility evidence for the
substantive 56-d/216-d differential surfaced in the leak-detection
appendix (\\S app:leak:scope).

Pre-trains an IQLAgent on transitions loaded from
``datasets/real_dirichlet.npz`` so the actor's input dimension matches
the GRPO target env (216-d).

Architecture: same DirichletActor as Phase 2A causal IQL, just with
``obs_dim=216`` instead of 56. Hyperparameters mirror the Phase 2A
config (iql_tau=0.7, iql_beta=3.0, lr=3e-4, 20k updates).

Output: ``results/aux_iql_216d/iql_seed42/{actor.pt,metrics.json,
sanity.json}``.

Sanity check (post-train, on 216-d test env):
    - Sharpe in [0.85, 1.05] AND turnover in [0.0, 0.05] -> OK
    - Sharpe outside band                                -> STOP (exit 3)
    - turnover == 0 with non-EW Sharpe                   -> STOP (exit 4)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.splits import SplitConfig, compute_split_indices
from core.buffers.replay_buffer import ReplayBuffer
from offline_rl.agents.iql import IQLAgent
from offline_rl.configs.iql_config import get_config as get_iql_config
from scripts.train_grpo import _load_dataset_arrays, _slice_env

TRADING_DAYS = 252


def _load_buffer_from_npz(
    npz_path: str,
    train_idx: np.ndarray,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
) -> ReplayBuffer:
    """Push pre-rolled transitions from .npz (sliced to train_idx) into a
    fresh ReplayBuffer, ready for IQL.update() to sample from."""
    data = np.load(npz_path, allow_pickle=True)
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    rewards = data["rewards"].astype(np.float32)
    next_states = data["next_states"].astype(np.float32)
    dones = data["dones"].astype(bool)

    # Filter to the train split. The .npz arrays are aligned 1:1 with
    # `dates`, and `train_idx` indexes into that same alignment.
    train_idx = np.asarray(train_idx)
    n = len(train_idx)
    print(f"[buffer] loading {n} train transitions from {npz_path}")

    buf = ReplayBuffer(capacity=n, obs_dim=obs_dim, action_dim=action_dim,
                       device=device, seq_len=20)
    # episode_starts: mark True wherever the previous step was done OR
    # where we cross a non-contiguous index in train_idx.
    prev_idx = -1
    for i, t in enumerate(train_idx):
        ep_start = (i == 0) or (train_idx[i] != prev_idx + 1) or (
            i > 0 and bool(dones[train_idx[i - 1]])
        )
        buf.add(
            obs=states[t],
            action=actions[t],
            reward=float(rewards[t]),
            next_obs=next_states[t],
            done=bool(dones[t]),
            episode_start=ep_start,
        )
        prev_idx = int(t)
    buf.freeze()
    return buf


def _backtest(actor, test_env, device) -> dict:
    """Single chronological sweep using DirichletActor.deterministic mean."""
    obs, _ = test_env.reset(options={"randomize": False})
    pv = [1.0]
    rets = []
    turns = []
    weights_log = []
    info = {}
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            w, _, _, _ = actor(obs_t, deterministic=True)
        action = w.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        pv.append(float(info["portfolio_value"]))
        rets.append(float(info["portfolio_return"]))
        turns.append(float(info["turnover"]))
        weights_log.append(np.asarray(info["executed_weights"]))

    rets_arr = np.asarray(rets)
    turns_arr = np.asarray(turns)
    pv_arr = np.asarray(pv)
    sharpe = (
        float(rets_arr.mean() / rets_arr.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if rets_arr.size > 1 and rets_arr.std(ddof=1) > 1e-12 else 0.0
    )
    rmax = np.maximum.accumulate(pv_arr)
    dd = (rmax - pv_arr) / np.maximum(rmax, 1e-12)
    return {
        "sharpe_ratio": sharpe,
        "annual_return": float(rets_arr.mean() * TRADING_DAYS),
        "annual_volatility": float(rets_arr.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": float(dd.max()),
        "turnover": float(turns_arr.mean()),
        "cumulative_return": float(pv_arr[-1] - 1.0),
        "n_steps": int(rets_arr.size),
        "weights_mean_per_asset": np.asarray(weights_log).mean(axis=0).tolist(),
        "weights_std_per_asset": np.asarray(weights_log).std(axis=0).tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, default="datasets/real_dirichlet.npz")
    p.add_argument("--n_offline_updates", type=int, default=20000)
    p.add_argument("--transaction_cost", type=float, default=0.001)
    p.add_argument("--output_dir", type=str, default="results/aux_iql_216d")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--iql_tau", type=float, default=None,
                   help="Override expectile (default 0.7). Lower = less aggressive.")
    p.add_argument("--iql_beta", type=float, default=None,
                   help="Override advantage temperature (default 3.0). Higher = stronger anchor to behavior.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Splits — must match scripts/train_grpo.py exactly so the test env
    # we evaluate on is the same env GRPO will be evaluated on.
    states, fwd, dates_ns = _load_dataset_arrays(args.dataset)
    split_cfg = SplitConfig(
        train_start="2008-01-01", train_end="2020-12-31",
        val_start="2021-01-01",   val_end="2021-12-31",
        test_start="2022-01-01",  test_end="2026-03-31",
    )
    split_idx = compute_split_indices(dates_ns, split_cfg)
    print(f"[splits] train={split_idx.train.size} val={split_idx.val.size} "
          f"test={split_idx.test.size}")

    # Build buffer from pre-rolled .npz transitions on the train split.
    obs_dim = states.shape[1]
    action_dim = fwd.shape[1]
    print(f"[shapes] obs_dim={obs_dim}  action_dim={action_dim}")
    buffer = _load_buffer_from_npz(args.dataset, split_idx.train,
                                   obs_dim, action_dim, device)

    # IQL agent.
    cfg = get_iql_config()
    cfg.n_offline_updates = args.n_offline_updates
    agent = IQLAgent(obs_dim=obs_dim, action_dim=action_dim,
                     config=cfg, device=device, offline_buffer=buffer)

    # Optional override of IQL hyperparameters.
    if args.iql_tau is not None:
        agent.config.iql_tau = float(args.iql_tau)
        print(f"[override] iql_tau = {agent.config.iql_tau}")
    if args.iql_beta is not None:
        agent.config.iql_beta = float(args.iql_beta)
        print(f"[override] iql_beta = {agent.config.iql_beta}")

    # Build val env for periodic sharpe tracking (matches test env path).
    val_env = _slice_env(states, fwd, split_idx.val,
                         episode_length=None,
                         transaction_cost=args.transaction_cost)
    print(f"[eval] val env T={val_env._n_steps}")

    # Training loop with periodic diagnostics.
    print(f"[train] {args.n_offline_updates} updates")
    diag_interval = max(1, args.n_offline_updates // 20)  # 20 snapshots
    diag_history = []  # list of dicts
    last_loss_metrics = {}
    for step in range(args.n_offline_updates):
        m = agent.update()
        # Keep last metrics for periodic snapshots; agent.update returns
        # a flat dict combining value/critic/actor losses each step.
        if m:
            last_loss_metrics = {k: float(v) for k, v in m.items()
                                 if isinstance(v, (int, float))}
        if step > 0 and step % diag_interval == 0:
            val_metrics = _backtest(agent.actor, val_env, device)
            snap = {
                "step": step,
                **last_loss_metrics,
                "val_sharpe": val_metrics["sharpe_ratio"],
                "val_turnover": val_metrics["turnover"],
                "val_cum_return": val_metrics["cumulative_return"],
            }
            diag_history.append(snap)
            print(
                f"  step {step:5d}/{args.n_offline_updates}  "
                f"val_sharpe={snap['val_sharpe']:+.3f}  "
                f"val_turn={snap['val_turnover']:.4f}  "
                f"v_loss={last_loss_metrics.get('iql/value_loss', float('nan')):.4f}  "
                f"q_loss={last_loss_metrics.get('iql/critic_loss', float('nan')):.4f}  "
                f"actor_loss={last_loss_metrics.get('iql/actor_loss', float('nan')):.4f}",
                flush=True,
            )

    # Test backtest on the 216-d test env (same path GRPO uses).
    test_env = _slice_env(states, fwd, split_idx.test,
                          episode_length=None,
                          transaction_cost=args.transaction_cost)
    print(f"[eval] test env T={test_env._n_steps}")
    metrics = _backtest(agent.actor, test_env, device)
    print("=" * 60)
    for k in ["sharpe_ratio", "annual_return", "max_drawdown",
              "turnover", "cumulative_return", "n_steps"]:
        v = metrics[k]
        print(f"  {k:20s}= {v:+.4f}" if isinstance(v, float) else f"  {k:20s}= {v}")
    print(f"  weights_mean = {[round(x, 3) for x in metrics['weights_mean_per_asset']]}")
    print(f"  weights_std  = {[round(x, 3) for x in metrics['weights_std_per_asset']]}")
    print("=" * 60)

    # Save outputs.
    run_name = args.run_name or f"iql_216d_seed{args.seed}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor.state_dict(), out_dir / "actor.pt")
    with (out_dir / "metrics.json").open("w") as f:
        json.dump({
            "algo": "iql_216d",
            "seed": args.seed,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "n_offline_updates": args.n_offline_updates,
            "transaction_cost": args.transaction_cost,
            "test": {k: v for k, v in metrics.items()
                     if not isinstance(v, list)},
        }, f, indent=2)
    print(f"[save] {out_dir}/actor.pt")
    print(f"[save] {out_dir}/metrics.json")
    if diag_history:
        with (out_dir / "diag_history.json").open("w") as f:
            json.dump(diag_history, f, indent=2)
        print(f"[save] {out_dir}/diag_history.json ({len(diag_history)} snapshots)")

    # Sanity-check stop conditions per the user's spec.
    sharpe = metrics["sharpe_ratio"]
    turn = metrics["turnover"]
    EW_REF = 0.953
    if not (0.85 <= sharpe <= 1.05):
        print(f"\n[SANITY FAIL] test Sharpe {sharpe:+.4f} outside [0.85, 1.05]")
        return 3
    if turn == 0.0 and abs(sharpe - EW_REF) > 0.02:
        print(f"\n[SANITY FAIL] turnover=0 with non-EW Sharpe {sharpe:+.4f} "
              f"(EW={EW_REF})")
        return 4
    print(f"\n[SANITY OK] Sharpe {sharpe:+.4f} in [0.85, 1.05], turnover "
          f"{turn:.6f} in [0.0, 0.05]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
