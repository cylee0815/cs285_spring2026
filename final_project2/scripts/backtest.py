"""CLI entrypoint for backtesting a trained IQL agent against baselines.

Usage
-----
    python scripts/backtest.py \
        --checkpoint checkpoints/iql.pt \
        --dataset datasets/test.npz \
        --device cuda

Outputs
-------
    results/<run_name>/metrics.csv
    results/<run_name>/tearsheet.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.iql import IQL
from env.portfolio_env import PortfolioEnv
from evaluation.backtest import run_backtest
from evaluation.baselines import (
    equal_weight_backtest,
    momentum_backtest,
    risk_parity_backtest,
)
from evaluation.metrics import compute_all_metrics, compute_max_drawdown, compute_rolling_sharpe
from utils.experiment_logger import RunLogger
from utils.seed import resolve_device, set_seed


# ------------------------------------------------------------------
# Agent loading
# ------------------------------------------------------------------


def load_agent(checkpoint_path: str, device: str) -> IQL:
    """Load a trained IQL agent from a checkpoint.

    The checkpoint is expected to contain keys:
    ``state_dim``, ``action_dim``, ``q_network``, ``value_network``,
    ``policy_network``, and optionally hyperparameters.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent = IQL(
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        hidden_sizes=ckpt.get("hidden_sizes", (256, 256)),
        gamma=ckpt.get("gamma", 0.99),
        tau=ckpt.get("tau", 0.7),
        beta=ckpt.get("beta", 3.0),
        device=device,
    )
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.value_network.load_state_dict(ckpt["value_network"])
    agent.policy_network.load_state_dict(ckpt["policy_network"])
    return agent


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def plot_tearsheet(
    results: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate a three-panel tearsheet and save to disk.

    Panel 1: equity curves for all strategies.
    Panel 2: drawdown curves.
    Panel 3: rolling Sharpe (window=63).
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # --- Panel 1: equity curves ---
    ax1 = axes[0]
    for name, res in results.items():
        ax1.plot(res["equity_curve"], label=name)
    ax1.set_ylabel("Equity")
    ax1.set_title("Equity Curves")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: drawdown ---
    ax2 = axes[1]
    for name, res in results.items():
        eq = np.asarray(res["equity_curve"], dtype=np.float64)
        running_max = np.maximum.accumulate(eq)
        dd = (running_max - eq) / np.where(running_max > 0, running_max, 1.0)
        ax2.plot(dd, label=name)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown Curves")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: rolling Sharpe ---
    ax3 = axes[2]
    for name, res in results.items():
        rs = compute_rolling_sharpe(res["portfolio_returns"], window=63)
        ax3.plot(rs, label=name)
    ax3.set_ylabel("Rolling Sharpe (63d)")
    ax3.set_xlabel("Time Step")
    ax3.set_title("Rolling Sharpe Ratio")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Tearsheet saved to {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a trained IQL agent.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pt file."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to test .npz dataset."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument(
        "--run_name",
        type=str,
        default="default",
        help="Sub-directory under results/ for outputs.",
    )
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Running on device: {device}")

    # --- Load agent ---
    agent = load_agent(args.checkpoint, device=device)

    # --- Load dataset ---
    data = np.load(args.dataset)
    states = data["states"].astype(np.float32)
    if "returns" not in data:
        raise ValueError(
            "Dataset must contain a 'returns' key with forward asset returns "
            "of shape (T, n_assets) for backtesting."
        )
    forward_returns = data["returns"].astype(np.float64)

    # --- IQL backtest ---
    env = PortfolioEnv(
        features=states,
        forward_returns=forward_returns.astype(np.float32),
        transaction_cost_lambda=args.transaction_cost,
    )
    iql_results = run_backtest(agent, env, device=device)

    # --- Baseline backtests ---
    ew_results = equal_weight_backtest(
        forward_returns, transaction_cost_lambda=args.transaction_cost
    )
    mom_results = momentum_backtest(
        forward_returns, transaction_cost_lambda=args.transaction_cost
    )
    rp_results = risk_parity_backtest(
        forward_returns, transaction_cost_lambda=args.transaction_cost
    )

    all_results = {
        "IQL": iql_results,
        "Equal Weight": ew_results,
        "Momentum": mom_results,
        "Risk Parity": rp_results,
    }

    # --- Compute metrics for baselines ---
    for name in ["Equal Weight", "Momentum", "Risk Parity"]:
        res = all_results[name]
        res["metrics"] = compute_all_metrics(
            res["portfolio_returns"], res["equity_curve"], res["weights"]
        )

    # --- Print metrics ---
    print("\n" + "=" * 70)
    print(f"{'Strategy':<15} {'AnnRet':>10} {'AnnVol':>10} {'Sharpe':>10} {'MDD':>10} {'Turnover':>10}")
    print("-" * 70)
    for name, res in all_results.items():
        m = res["metrics"]
        print(
            f"{name:<15} {m['annual_return']:>+10.4f} {m['annual_volatility']:>10.4f} "
            f"{m['sharpe_ratio']:>10.4f} {m['max_drawdown']:>10.4f} {m['turnover']:>10.4f}"
        )
    print("=" * 70)

    # --- Save results ---
    output_dir = _PROJECT_ROOT / "results" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.csv
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["strategy"] + list(next(iter(all_results.values()))["metrics"].keys())
        writer.writerow(header)
        for name, res in all_results.items():
            row = [name] + [res["metrics"][k] for k in header[1:]]
            writer.writerow(row)
    print(f"Metrics saved to {csv_path}")

    # Save curves and final metrics via RunLogger
    iql_metrics = all_results["IQL"]["metrics"]
    iql_returns = all_results["IQL"]["portfolio_returns"]
    iql_weights = all_results["IQL"]["weights"]
    equity_curve = all_results["IQL"]["equity_curve"]
    cumulative_return = np.cumprod(1 + iql_returns) - 1

    run_logger = RunLogger(
        run_dir=output_dir,
        run_name=args.run_name,
        config={
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "seed": args.seed,
            "transaction_cost": args.transaction_cost,
        },
        wandb_enabled=args.wandb,
    )
    run_logger.log_curves(equity_curve, cumulative_return)
    run_logger.log_backtest_arrays(
        portfolio_returns=iql_returns,
        weights=iql_weights,
    )
    run_logger.log_final_metrics({
        "sharpe": iql_metrics["sharpe_ratio"],
        "max_drawdown": iql_metrics["max_drawdown"],
        "turnover": iql_metrics["turnover"],
        "final_cumulative_return": iql_metrics["cumulative_return"],
    })
    run_logger.close()

    # tearsheet.png
    tearsheet_path = output_dir / "tearsheet.png"
    plot_tearsheet(all_results, tearsheet_path)


if __name__ == "__main__":
    main()
