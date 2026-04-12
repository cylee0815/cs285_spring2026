"""Run ablation experiments over a hyperparameter grid.

Each hyperparameter combination trains an IQL agent on the **train split
only**, runs periodic validation on the held-out validation split, saves
the best checkpoint by validation Sharpe, and then reports final metrics
on the **test split** (once, using the best checkpoint).

Usage
-----
    uv run python scripts/run_ablation.py --config configs/experiments.yaml
    uv run python scripts/run_ablation.py --config configs/experiments.yaml --dry-run
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import yaml

from algorithms.iql import IQL
from data.splits import (
    DEFAULT_SPLIT,
    SplitConfig,
    compute_split_indices,
)
from evaluation.baselines import (
    equal_weight_backtest,
    momentum_backtest,
    risk_parity_backtest,
)
from evaluation.metrics import compute_all_metrics
from evaluation.split_backtest import backtest_on_indices
from training.train_iql import train_iql
from utils.experiment_logger import RunLogger
from utils.replay_buffer import ReplayBuffer
from utils.seed import resolve_device, set_seed


def _load_experiment_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_split_cfg(raw: dict[str, Any] | None) -> SplitConfig:
    if not raw:
        return DEFAULT_SPLIT
    merged = {
        "train_start": raw.get("train_start", DEFAULT_SPLIT.train_start),
        "train_end": raw.get("train_end", DEFAULT_SPLIT.train_end),
        "val_start": raw.get("val_start", DEFAULT_SPLIT.val_start),
        "val_end": raw.get("val_end", DEFAULT_SPLIT.val_end),
        "test_start": raw.get("test_start", DEFAULT_SPLIT.test_start),
        "test_end": raw.get("test_end", DEFAULT_SPLIT.test_end),
    }
    return SplitConfig(**merged)


def _load_dataset_with_returns(dataset_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load states, forward_returns, and dates arrays for splitting + backtesting."""
    data = np.load(dataset_path, allow_pickle=True)
    states = data["states"].astype(np.float32)

    if "forward_returns" in data.files:
        forward_returns = data["forward_returns"].astype(np.float32)
        if forward_returns.shape[0] != states.shape[0]:
            forward_returns = forward_returns[: states.shape[0]]
    else:
        print("[warn] dataset has no 'forward_returns' — backtest values will be "
              "a zero-return proxy. Rebuild with scripts/build_real_dataset.py.")
        forward_returns = np.zeros_like(data["actions"], dtype=np.float32)

    if "dates" in data.files:
        dates = data["dates"].astype(np.int64)
    else:
        print("[warn] dataset has no 'dates' — synthesising a business-day axis.")
        import pandas as pd
        synthetic = pd.bdate_range(start="2008-01-02", periods=states.shape[0])
        dates = synthetic.values.astype("datetime64[ns]").astype(np.int64)
    return states, forward_returns, dates


def run_single_experiment(
    experiment_id: str,
    base_cfg: dict[str, Any],
    split_cfg: SplitConfig,
    expectile: float,
    beta: float,
    transaction_cost: float,
    output_dir: Path,
    device: str,
    wandb_enabled: bool = False,
) -> dict[str, float]:
    """Train + validate + test a single hyperparameter configuration."""
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_id}")
    print(f"  expectile={expectile}, beta={beta}, tc={transaction_cost}")
    print(f"{'='*60}")

    set_seed(base_cfg.get("seed", 42))

    dataset_path = base_cfg["dataset"]
    data = np.load(dataset_path)
    state_dim = data["states"].shape[1]
    action_dim = data["actions"].shape[1]

    # Build the date-based split — shared across every ablation run.
    states, forward_returns, dates_ns = _load_dataset_with_returns(dataset_path)
    split_idx = compute_split_indices(dates_ns, split_cfg)
    # Hard assertions on split integrity — identical to scripts/train.py.
    assert split_idx.train.size > 0 and split_idx.val.size > 0 and split_idx.test.size > 0
    for name, idx in (("train", split_idx.train),
                      ("val", split_idx.val),
                      ("test", split_idx.test)):
        assert np.all(np.diff(dates_ns[idx]) >= 0), (
            f"Split '{name}' dates not monotonically non-decreasing."
        )
    assert int(dates_ns[split_idx.train].max()) < int(dates_ns[split_idx.val].min())
    assert int(dates_ns[split_idx.val].max()) < int(dates_ns[split_idx.test].min())
    _all = np.concatenate([split_idx.train, split_idx.val, split_idx.test])
    assert _all.size == np.unique(_all).size, "Split indices overlap."
    print(f"  splits → train={split_idx.train.size}, "
          f"val={split_idx.val.size}, test={split_idx.test.size}")

    exp_dir = output_dir / experiment_id
    exp_config = {
        "experiment_id": experiment_id,
        "expectile": expectile,
        "beta": beta,
        "transaction_cost": transaction_cost,
        **{k: v for k, v in base_cfg.items() if k not in ("dataset",)},
        "dataset": str(dataset_path),
        "split": split_cfg.__dict__,
    }
    run_logger = RunLogger(
        run_dir=exp_dir,
        run_name=experiment_id,
        config=exp_config,
        wandb_enabled=wandb_enabled,
    )

    buffer = ReplayBuffer(dataset_path, device=device, indices=split_idx.train)
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

    ckpt_path = exp_dir / "iql.pt"
    best_ckpt_path = exp_dir / "best_model.pt"
    best_state = {"sharpe": float("-inf"), "step": -1}

    def _save_ckpt(path: Path) -> None:
        torch.save({
            "state_dim": state_dim,
            "action_dim": action_dim,
            "q_network": agent.q_network.state_dict(),
            "value_network": agent.value_network.state_dict(),
            "policy_network": agent.policy_network.state_dict(),
            "gamma": base_cfg.get("gamma", 0.99),
            "tau": expectile,
            "beta": beta,
        }, path)

    def _validation_fn(step: int) -> None:
        res = backtest_on_indices(
            agent=agent,
            states=states,
            forward_returns=forward_returns,
            indices=split_idx.val,
            transaction_cost_lambda=transaction_cost,
            device=device,
        )
        m = res["metrics"]
        sharpe = float(m["sharpe_ratio"])
        ann_ret = float(m["annual_return"])
        mdd = float(m["max_drawdown"])
        cum_ret = float(m["cumulative_return"])
        turnover = float(m["turnover"])

        print(f"  [val @ step {step:>7d}] sharpe={sharpe:+.4f}  "
              f"ann_ret={ann_ret:+.4f}  mdd={mdd:.4f}  "
              f"turnover={turnover:.4f}  cum_ret={cum_ret:+.4f}")
        run_logger.log_validation_metrics(step, {
            "sharpe_ratio": sharpe,
            "annual_return": ann_ret,
            "max_drawdown": mdd,
            "turnover": turnover,
            "cumulative_return": cum_ret,
        })
        val_weights = np.asarray(res["weights"])
        val_dates = dates_ns[split_idx.val][:val_weights.shape[0]].astype(np.int64)
        run_logger.log_validation_curves(
            step=step,
            equity_curve=np.asarray(res["equity_curve"]),
            portfolio_returns=np.asarray(res["portfolio_returns"]),
            weights=val_weights,
            dates=val_dates,
        )
        if sharpe > best_state["sharpe"]:
            best_state["sharpe"] = sharpe
            best_state["step"] = step
            _save_ckpt(best_ckpt_path)
            with open(best_ckpt_path.with_suffix(".metrics.json"), "w") as f:
                json.dump({
                    "step": step,
                    "val_sharpe": sharpe,
                    "val_annual_return": ann_ret,
                    "val_max_drawdown": mdd,
                    "val_cumulative_return": cum_ret,
                }, f, indent=2)

    log_interval = base_cfg.get("log_interval", 1000)
    val_interval = base_cfg.get("val_interval", 5000)
    train_iql(
        agent=agent,
        buffer=buffer,
        total_steps=base_cfg.get("steps", 100_000),
        batch_size=base_cfg.get("batch_size", 256),
        log_interval=log_interval,
        log_fn=run_logger.make_log_fn(log_interval),
        validation_fn=_validation_fn,
        validation_interval=val_interval,
    )

    # Save final-step checkpoint too.
    _save_ckpt(ckpt_path)
    if best_state["step"] < 0:
        _save_ckpt(best_ckpt_path)

    # Reload best checkpoint and run test evaluation once.
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.value_network.load_state_dict(ckpt["value_network"])
    agent.policy_network.load_state_dict(ckpt["policy_network"])

    iql_results = backtest_on_indices(
        agent=agent,
        states=states,
        forward_returns=forward_returns,
        indices=split_idx.test,
        transaction_cost_lambda=transaction_cost,
        device=device,
    )

    # Baselines on the test window (apples-to-apples comparison).
    test_returns = forward_returns[split_idx.test].astype(np.float64)
    ew_results = equal_weight_backtest(test_returns, transaction_cost_lambda=transaction_cost)
    mom_results = momentum_backtest(test_returns, transaction_cost_lambda=transaction_cost)
    rp_results = risk_parity_backtest(test_returns, transaction_cost_lambda=transaction_cost)
    for name, res in [("Equal Weight", ew_results), ("Momentum", mom_results), ("Risk Parity", rp_results)]:
        res["metrics"] = compute_all_metrics(res["portfolio_returns"], res["equity_curve"], res["weights"])

    all_results = {
        "IQL": iql_results,
        "Equal Weight": ew_results,
        "Momentum": mom_results,
        "Risk Parity": rp_results,
    }

    # metrics.csv (legacy layout, now explicitly test-period).
    csv_path = exp_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["strategy"] + list(next(iter(all_results.values()))["metrics"].keys())
        writer.writerow(header)
        for name, res in all_results.items():
            row = [name] + [res["metrics"][k] for k in header[1:]]
            writer.writerow(row)

    iql_metrics = iql_results["metrics"]
    portfolio_returns = iql_results["portfolio_returns"]
    equity_curve = iql_results["equity_curve"]
    cumulative_return = np.cumprod(1 + portfolio_returns) - 1

    test_weights = np.asarray(iql_results["weights"])
    test_dates = dates_ns[split_idx.test][:test_weights.shape[0]].astype(np.int64)

    run_logger.log_curves(equity_curve, cumulative_return)
    run_logger.log_backtest_arrays(
        portfolio_returns=portfolio_returns,
        weights=test_weights,
        dates=test_dates,
    )
    run_logger.log_test_artifacts(
        metrics={
            "annual_return": float(iql_metrics["annual_return"]),
            "sharpe_ratio": float(iql_metrics["sharpe_ratio"]),
            "max_drawdown": float(iql_metrics["max_drawdown"]),
            "turnover": float(iql_metrics["turnover"]),
            "cumulative_return": float(iql_metrics["cumulative_return"]),
        },
        equity_curve=np.asarray(equity_curve),
        portfolio_returns=np.asarray(portfolio_returns),
        weights=test_weights,
        dates=test_dates,
    )
    run_logger.log_final_metrics({
        "best_val_sharpe": best_state["sharpe"],
        "best_val_step": best_state["step"],
        "test_sharpe": iql_metrics["sharpe_ratio"],
        "test_annual_return": iql_metrics["annual_return"],
        "test_annual_volatility": iql_metrics["annual_volatility"],
        "test_max_drawdown": iql_metrics["max_drawdown"],
        "test_turnover": iql_metrics["turnover"],
        "test_cumulative_return": iql_metrics["cumulative_return"],
        "ew_sharpe": ew_results["metrics"]["sharpe_ratio"],
        "mom_sharpe": mom_results["metrics"]["sharpe_ratio"],
        "rp_sharpe": rp_results["metrics"]["sharpe_ratio"],
    })

    np.save(exp_dir / "ew_equity_curve.npy", ew_results["equity_curve"])
    np.save(exp_dir / "mom_equity_curve.npy", mom_results["equity_curve"])
    np.save(exp_dir / "rp_equity_curve.npy", rp_results["equity_curve"])

    run_logger.close()

    print(f"  best val Sharpe={best_state['sharpe']:+.4f} @ step {best_state['step']}")
    print(f"  TEST Sharpe={iql_metrics['sharpe_ratio']:+.4f}, "
          f"MDD={iql_metrics['max_drawdown']:.4f}, "
          f"AnnRet={iql_metrics['annual_return']:+.4f}")

    return {
        "experiment_id": experiment_id,
        "expectile": expectile,
        "beta": beta,
        "transaction_cost": transaction_cost,
        "best_val_sharpe": best_state["sharpe"],
        "best_val_step": best_state["step"],
        **{f"test_{k}": v for k, v in iql_metrics.items()},
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
    split_cfg = _resolve_split_cfg(cfg.get("split"))
    grid = cfg.get("grid", {})

    expectiles = grid.get("expectile", [0.7])
    betas = grid.get("beta", [3.0])
    transaction_costs = grid.get("transaction_cost", [0.001])

    combinations = list(itertools.product(expectiles, betas, transaction_costs))
    print(f"Total experiments: {len(combinations)}")

    if args.dry_run:
        for i, (exp, beta, tc) in enumerate(combinations):
            eid = f"iql_exp_tau{exp}_beta{beta}_tc{tc}"
            print(f"  [{i+1}/{len(combinations)}] {eid}")
        return

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    summary_rows = []

    for i, (exp, beta, tc) in enumerate(combinations):
        eid = f"iql_exp_tau{exp}_beta{beta}_tc{tc}"
        print(f"\n[{i+1}/{len(combinations)}]")
        row = run_single_experiment(
            experiment_id=eid,
            base_cfg=base_cfg,
            split_cfg=split_cfg,
            expectile=exp,
            beta=beta,
            transaction_cost=tc,
            output_dir=output_dir,
            device=device,
            wandb_enabled=args.wandb,
        )
        summary_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    if summary_rows:
        fieldnames: list[str] = []
        for row in summary_rows:
            for k in row.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nAblation summary saved to {summary_path}")

    print(f"\nAll {len(combinations)} experiments complete.")


if __name__ == "__main__":
    main()
