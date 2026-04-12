"""Run ablation experiments over a hyperparameter grid.

Usage
-----
    python scripts/run_ablation.py --config configs/experiments.yaml
    python scripts/run_ablation.py --config configs/experiments.yaml --dry-run

Each combination of (expectile, beta, transaction_cost) trains a model,
runs a backtest, and stores results in results/ablation/<experiment_id>/.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import yaml

from algorithms.iql import IQL
from env.portfolio_env import PortfolioEnv
from evaluation.backtest import run_backtest
from evaluation.baselines import equal_weight_backtest, momentum_backtest, risk_parity_backtest
from evaluation.metrics import compute_all_metrics
from training.train_iql import train_iql
from utils.experiment_logger import RunLogger
from utils.replay_buffer import ReplayBuffer
from utils.seed import resolve_device, set_seed


def _load_experiment_config(path: str) -> dict:
    """Load experiment grid config."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def run_single_experiment(
    experiment_id: str,
    base_cfg: dict,
    expectile: float,
    beta: float,
    transaction_cost: float,
    output_dir: Path,
    device: str,
    wandb_enabled: bool = False,
) -> dict[str, float]:
    """Train + backtest a single hyperparameter configuration.

    Returns
    -------
    Dict with experiment_id, hyperparams, and all metrics.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_id}")
    print(f"  expectile={expectile}, beta={beta}, tc={transaction_cost}")
    print(f"{'='*60}")

    set_seed(base_cfg.get("seed", 42))

    dataset_path = base_cfg["dataset"]
    data = np.load(dataset_path)
    state_dim = data["states"].shape[1]
    action_dim = data["actions"].shape[1]

    exp_dir = output_dir / experiment_id
    exp_config = {
        "experiment_id": experiment_id,
        "expectile": expectile,
        "beta": beta,
        "transaction_cost": transaction_cost,
        **{k: v for k, v in base_cfg.items() if k not in ("dataset",)},
        "dataset": str(dataset_path),
    }

    # Initialize experiment logger (W&B + local)
    run_logger = RunLogger(
        run_dir=exp_dir,
        run_name=experiment_id,
        config=exp_config,
        wandb_enabled=wandb_enabled,
    )

    buffer = ReplayBuffer(dataset_path, device=device)
    agent = IQL(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=base_cfg.get("lr", 3e-4),
        gamma=base_cfg.get("gamma", 0.99),
        tau=expectile,
        beta=beta,
        polyak=base_cfg.get("polyak", 0.005),
        device=device,
    )

    log_interval = base_cfg.get("log_interval", 1000)
    train_iql(
        agent=agent,
        buffer=buffer,
        total_steps=base_cfg.get("steps", 100_000),
        batch_size=base_cfg.get("batch_size", 256),
        log_interval=log_interval,
        log_fn=run_logger.make_log_fn(log_interval),
    )

    # Save checkpoint
    ckpt_path = exp_dir / "iql.pt"
    torch.save({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "q_network": agent.q_network.state_dict(),
        "value_network": agent.value_network.state_dict(),
        "policy_network": agent.policy_network.state_dict(),
        "gamma": base_cfg.get("gamma", 0.99),
        "tau": expectile,
        "beta": beta,
    }, ckpt_path)

    # Backtest
    states = data["states"].astype(np.float32)
    if "returns" in data:
        forward_returns = data["returns"].astype(np.float32)
    else:
        forward_returns = data["actions"].astype(np.float32) * 0.01

    env = PortfolioEnv(
        features=states,
        forward_returns=forward_returns,
        transaction_cost_lambda=transaction_cost,
    )
    iql_results = run_backtest(agent, env, device=device)

    # Baselines
    fr64 = forward_returns.astype(np.float64)
    ew_results = equal_weight_backtest(fr64, transaction_cost_lambda=transaction_cost)
    mom_results = momentum_backtest(fr64, transaction_cost_lambda=transaction_cost)
    rp_results = risk_parity_backtest(fr64, transaction_cost_lambda=transaction_cost)

    for name, res in [("Equal Weight", ew_results), ("Momentum", mom_results), ("Risk Parity", rp_results)]:
        res["metrics"] = compute_all_metrics(res["portfolio_returns"], res["equity_curve"], res["weights"])

    all_results = {
        "IQL": iql_results,
        "Equal Weight": ew_results,
        "Momentum": mom_results,
        "Risk Parity": rp_results,
    }

    # Save metrics.csv (legacy format, kept for compatibility)
    csv_path = exp_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["strategy"] + list(next(iter(all_results.values()))["metrics"].keys())
        writer.writerow(header)
        for name, res in all_results.items():
            row = [name] + [res["metrics"][k] for k in header[1:]]
            writer.writerow(row)

    # Log curves and final metrics via RunLogger
    iql_metrics = iql_results["metrics"]
    portfolio_returns = iql_results["portfolio_returns"]
    equity_curve = iql_results["equity_curve"]
    cumulative_return = np.cumprod(1 + portfolio_returns) - 1

    run_logger.log_curves(equity_curve, cumulative_return)
    run_logger.log_final_metrics({
        "sharpe": iql_metrics["sharpe_ratio"],
        "max_drawdown": iql_metrics["max_drawdown"],
        "turnover": iql_metrics["turnover"],
        "final_cumulative_return": iql_metrics["cumulative_return"],
    })
    run_logger.close()

    print(f"  Sharpe={iql_metrics['sharpe_ratio']:.4f}, "
          f"MDD={iql_metrics['max_drawdown']:.4f}, "
          f"AnnRet={iql_metrics['annual_return']:.4f}")

    return {
        "experiment_id": experiment_id,
        "expectile": expectile,
        "beta": beta,
        "transaction_cost": transaction_cost,
        **iql_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml",
                        help="Path to experiment grid config.")
    parser.add_argument("--output_dir", type=str, default="results/ablation",
                        help="Root directory for ablation results.")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiment grid without running.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    args = parser.parse_args()

    cfg = _load_experiment_config(args.config)
    base_cfg = cfg.get("base", {})
    grid = cfg.get("grid", {})

    expectiles = grid.get("expectile", [0.7])
    betas = grid.get("beta", [3.0])
    transaction_costs = grid.get("transaction_cost", [0.001])

    combinations = list(itertools.product(expectiles, betas, transaction_costs))
    print(f"Total experiments: {len(combinations)}")

    if args.dry_run:
        for i, (exp, beta, tc) in enumerate(combinations):
            eid = f"exp_{exp}_{beta}_{tc}"
            print(f"  [{i+1}/{len(combinations)}] {eid}")
        return

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    summary_rows = []

    for i, (exp, beta, tc) in enumerate(combinations):
        eid = f"exp_{exp}_{beta}_{tc}"
        print(f"\n[{i+1}/{len(combinations)}]")
        row = run_single_experiment(
            experiment_id=eid,
            base_cfg=base_cfg,
            expectile=exp,
            beta=beta,
            transaction_cost=tc,
            output_dir=output_dir,
            device=device,
            wandb_enabled=args.wandb,
        )
        summary_rows.append(row)

    # Write summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nAblation summary saved to {summary_path}")

    print(f"\nAll {len(combinations)} experiments complete.")


if __name__ == "__main__":
    main()
