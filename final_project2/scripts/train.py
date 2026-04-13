"""CLI entrypoint for IQL training with date-based train/val/test splits.

The dataset is split by calendar date into three strictly non-overlapping
windows (configurable via ``--split_config`` or the defaults in
:mod:`data.splits`):

* training draws minibatches only from the train window,
* every ``--val_interval`` steps we run a full backtest on the validation
  window, log the metrics, and save a ``best_model.pt`` checkpoint whenever
  validation Sharpe improves,
* after training finishes we load that best checkpoint and run a single
  backtest on the test window, saving the artifacts to ``results/test/``.

Usage
-----
    uv run python scripts/train.py \\
        --dataset datasets/real_dirichlet.npz \\
        --steps 100000 --val_interval 5000

    uv run python scripts/train.py --config configs/iql_default.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so bare imports work.
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
    SplitIndices,
    compute_split_indices,
)
from evaluation.split_backtest import backtest_on_indices
from training.train_iql import train_iql
from utils.experiment_logger import RunLogger
from utils.replay_buffer import ReplayBuffer
from utils.seed import resolve_device, set_seed


def _load_yaml_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_split_cfg(raw: dict[str, Any] | None) -> SplitConfig:
    """Build a :class:`SplitConfig` from a CLI/YAML dict, defaulting to
    :data:`DEFAULT_SPLIT` for any missing field."""
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


def _check_split_is_valid(split_idx: SplitIndices, dates_ns: np.ndarray) -> None:
    """Safety assertions that the split is strictly time-ordered and non-overlapping.

    These are redundant with :func:`compute_split_indices`' internal checks
    but are re-run here so any future regression in the splitting logic
    fails loudly at the training entrypoint rather than silently leaking
    data across the train/val/test boundary.
    """
    # 1. Non-empty.
    assert split_idx.train.size > 0, "Train split is empty."
    assert split_idx.val.size > 0, "Validation split is empty."
    assert split_idx.test.size > 0, "Test split is empty."

    # 2. Date monotonicity within each split — the dataset must be
    #    date-sorted and we must not have shuffled it. This protects
    #    against anyone introducing a shuffle step upstream.
    for name, idx in (("train", split_idx.train),
                      ("val", split_idx.val),
                      ("test", split_idx.test)):
        d = dates_ns[idx]
        assert np.all(np.diff(d) >= 0), (
            f"Dates for split '{name}' are not monotonically non-decreasing — "
            f"the dataset must be sorted by date."
        )

    # 3. Strict time-ordering across splits.
    train_max = int(dates_ns[split_idx.train].max())
    val_min = int(dates_ns[split_idx.val].min())
    val_max = int(dates_ns[split_idx.val].max())
    test_min = int(dates_ns[split_idx.test].min())
    assert train_max < val_min, (
        f"Train dates overlap validation: max train={train_max}, min val={val_min}"
    )
    assert val_max < test_min, (
        f"Validation dates overlap test: max val={val_max}, min test={test_min}"
    )

    # 4. Pairwise index disjointness.
    all_idx = np.concatenate([split_idx.train, split_idx.val, split_idx.test])
    assert all_idx.size == np.unique(all_idx).size, "Split indices overlap."


def _load_dataset_with_returns(dataset_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``states``, ``forward_returns``, ``dates`` arrays required for splits.

    Falls back to a synthetic date axis for legacy datasets that lack dates —
    this preserves backwards compatibility for the old synthetic datasets
    but issues a loud warning so users don't accidentally evaluate on fake
    calendar boundaries.
    """
    data = np.load(dataset_path, allow_pickle=True)
    states = data["states"].astype(np.float32)

    if "forward_returns" in data.files:
        forward_returns = data["forward_returns"].astype(np.float32)
        # Must align with states (one forward-return row per transition).
        # For real datasets built from price data, the two arrays have the
        # same length T because ``build_features`` already trims the
        # undefined final row.
        if forward_returns.shape[0] != states.shape[0]:
            forward_returns = forward_returns[: states.shape[0]]
    else:
        # Legacy synthetic dataset: reward was the direct dot product with
        # a per-step return proxy. Fall back to actions-as-returns so the
        # backtest helper can still run, but this is only meaningful on
        # real data.
        print("[warn] dataset has no 'forward_returns' array — falling back "
              "to a zero-return proxy. Backtest values are meaningless.")
        forward_returns = np.zeros_like(data["actions"], dtype=np.float32)

    if "dates" in data.files:
        dates = data["dates"].astype(np.int64)
    else:
        print("[warn] dataset has no 'dates' array — synthesising a business-day "
              "axis starting 2008-01-02. Legacy compatibility only; real runs "
              "must use a date-aware dataset.")
        import pandas as pd
        synthetic = pd.bdate_range(start="2008-01-02", periods=states.shape[0])
        dates = synthetic.values.astype("datetime64[ns]").astype(np.int64)

    return states, forward_returns, dates


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IQL with date-based splits.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .npz dataset.")
    parser.add_argument("--steps", type=int, default=None, help="Number of gradient steps.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--expectile", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--polyak", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=None,
                        help="Run validation every N gradient steps.")
    parser.add_argument("--transaction_cost", type=float, default=None,
                        help="Transaction-cost lambda used during back-tests.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help=("Directory to save model checkpoints (best_model.pt + final.pt). "
              "Defaults to <runs_dir>/<run_name>/<timestamp>/checkpoints/."),
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Root directory for canonical per-experiment artifact mirrors.",
    )
    parser.add_argument(
        "--runs_dir", type=str, default="runs",
        help="Root directory for per-run timestamped local logs (runs/<name>/<ts>/).",
    )
    parser.add_argument("--run_name", type=str, default="iql_train",
                        help="Sub-directory under runs_dir for this run.")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    args = parser.parse_args()

    defaults = {
        "dataset": "datasets/real_dirichlet.npz",
        "steps": 20_000,
        "batch_size": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "expectile": 0.7,
        "beta": 3.0,
        "polyak": 0.005,
        "seed": 42,
        "device": "auto",
        "log_interval": 200,
        "val_interval": 1000,
        "transaction_cost": 0.001,
    }
    yaml_cfg: dict[str, Any] = {}
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        defaults.update({k: v for k, v in yaml_cfg.items() if k != "split"})
    for key in [
        "dataset", "steps", "batch_size", "lr", "gamma", "expectile",
        "beta", "polyak", "seed", "device", "log_interval",
        "val_interval", "transaction_cost",
    ]:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            defaults[key] = cli_val

    cfg = argparse.Namespace(**defaults)
    split_cfg = _resolve_split_cfg(yaml_cfg.get("split") if yaml_cfg else None)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # --- Dataset + splits -------------------------------------------------
    states, forward_returns, dates_ns = _load_dataset_with_returns(cfg.dataset)
    data = np.load(cfg.dataset)
    state_dim = data["states"].shape[1]
    action_dim = data["actions"].shape[1]
    print(
        f"Dataset: {states.shape[0]} transitions, "
        f"state_dim={state_dim}, action_dim={action_dim}"
    )

    split_idx = compute_split_indices(dates_ns, split_cfg)
    _check_split_is_valid(split_idx, dates_ns)
    print(f"Splits → train={split_idx.train.size}, "
          f"val={split_idx.val.size}, test={split_idx.test.size}")

    # --- Agent + buffer (train only) -------------------------------------
    buffer = ReplayBuffer(cfg.dataset, device=device, indices=split_idx.train)
    agent = IQL(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        tau=cfg.expectile,
        beta=cfg.beta,
        polyak=cfg.polyak,
        device=device,
    )

    # --- Run directory + logger ------------------------------------------
    # Each run lives at runs/<run_name>/<timestamp>/ so repeat invocations
    # do not overwrite prior logs. Canonical mirror dirs (results/validation,
    # results/test) remain at their stable locations for downstream tooling.
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / args.run_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    abs_run_dir = run_dir.resolve()
    print(f"[INFO] Saving logs to: {abs_run_dir}")

    wandb_name = f"iql_{cfg.expectile}_{cfg.beta}_{cfg.transaction_cost}"
    logged_cfg = {
        **vars(cfg),
        "split": split_cfg.__dict__,
        "run_name": args.run_name,
        "wandb_name": wandb_name,
        "run_dir": str(abs_run_dir),
        "timestamp": timestamp,
    }
    run_logger = RunLogger(
        run_dir=run_dir,
        run_name=wandb_name,
        config=logged_cfg,
        wandb_enabled=args.wandb,
        capture_stdout=True,
    )

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pt"
    final_ckpt_path = ckpt_dir / "iql.pt"

    best_state: dict[str, float] = {"sharpe": float("-inf"), "step": -1}

    def _save_checkpoint(path: Path) -> None:
        torch.save({
            "state_dim": state_dim,
            "action_dim": action_dim,
            "q_network": agent.q_network.state_dict(),
            "value_network": agent.value_network.state_dict(),
            "policy_network": agent.policy_network.state_dict(),
            "gamma": cfg.gamma,
            "tau": cfg.expectile,
            "beta": cfg.beta,
        }, path)

    # --- Validation callback --------------------------------------------
    def _validation_fn(step: int) -> None:
        res = backtest_on_indices(
            agent=agent,
            states=states,
            forward_returns=forward_returns,
            indices=split_idx.val,
            transaction_cost_lambda=cfg.transaction_cost,
            device=device,
        )
        val_metrics = res["metrics"]
        sharpe = float(val_metrics["sharpe_ratio"])
        ann_ret = float(val_metrics["annual_return"])
        max_dd = float(val_metrics["max_drawdown"])
        cum_ret = float(val_metrics["cumulative_return"])
        turnover = float(val_metrics["turnover"])

        print(f"[val @ step {step:>7d}] sharpe={sharpe:+.4f}  "
              f"ann_ret={ann_ret:+.4f}  mdd={max_dd:.4f}  "
              f"turnover={turnover:.4f}  cum_ret={cum_ret:+.4f}")

        run_logger.log_validation_metrics(step, {
            "sharpe_ratio": sharpe,
            "annual_return": ann_ret,
            "max_drawdown": max_dd,
            "turnover": turnover,
            "cumulative_return": cum_ret,
        })

        # Backtest yields one weight/return row per simulated step.
        # ``backtest_on_indices`` steps an environment of length
        # ``len(split_idx.val)`` starting at index 0 and consumes
        # ``T - 1`` transitions, so the dates align with the first
        # ``len(weights)`` entries of the validation-indexed date slice.
        weights_np = np.asarray(res["weights"])
        T_val = weights_np.shape[0]
        val_dates = dates_ns[split_idx.val][:T_val].astype(np.int64)

        run_logger.log_validation_curves(
            step=step,
            equity_curve=np.asarray(res["equity_curve"]),
            portfolio_returns=np.asarray(res["portfolio_returns"]),
            weights=weights_np,
            dates=val_dates,
        )
        # Mirror canonical artifacts to results/validation/<run_name>/
        # so analysis tooling can find them at a stable location.
        global_val_dir = Path(args.results_dir) / "validation" / args.run_name
        global_val_dir.mkdir(parents=True, exist_ok=True)
        np.save(global_val_dir / "equity_curve.npy", np.asarray(res["equity_curve"]))
        np.save(global_val_dir / "portfolio_returns.npy",
                np.asarray(res["portfolio_returns"]))
        np.save(global_val_dir / "weights.npy", weights_np)
        np.save(global_val_dir / "dates.npy", val_dates)

        if sharpe > best_state["sharpe"]:
            best_state["sharpe"] = sharpe
            best_state["step"] = step
            _save_checkpoint(best_ckpt_path)
            # Also save the validation metrics at the best step for audit.
            import json
            with open(best_ckpt_path.with_suffix(".metrics.json"), "w") as f:
                json.dump({
                    "step": step,
                    "val_sharpe": sharpe,
                    "val_annual_return": ann_ret,
                    "val_max_drawdown": max_dd,
                    "val_cumulative_return": cum_ret,
                }, f, indent=2)
            print(f"  ★ new best val Sharpe — checkpoint → {best_ckpt_path}")

    # --- Training loop ---------------------------------------------------
    train_iql(
        agent=agent,
        buffer=buffer,
        total_steps=cfg.steps,
        batch_size=cfg.batch_size,
        log_interval=cfg.log_interval,
        log_fn=run_logger.make_log_fn(cfg.log_interval),
        validation_fn=_validation_fn,
        validation_interval=cfg.val_interval,
    )

    # Save the last-step checkpoint too (useful as a baseline).
    _save_checkpoint(final_ckpt_path)
    print(f"Final checkpoint saved to {final_ckpt_path}")

    if best_state["step"] < 0:
        print("[warn] no validation evaluation was performed — using final checkpoint for test.")
        _save_checkpoint(best_ckpt_path)

    # --- Test evaluation (once, on best checkpoint) ----------------------
    print(f"\nLoading best checkpoint (step={best_state['step']}, "
          f"val Sharpe={best_state['sharpe']:+.4f}) → test evaluation.")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    agent.q_network.load_state_dict(ckpt["q_network"])
    agent.value_network.load_state_dict(ckpt["value_network"])
    agent.policy_network.load_state_dict(ckpt["policy_network"])

    test_res = backtest_on_indices(
        agent=agent,
        states=states,
        forward_returns=forward_returns,
        indices=split_idx.test,
        transaction_cost_lambda=cfg.transaction_cost,
        device=device,
    )
    test_metrics = test_res["metrics"]
    test_sharpe = float(test_metrics["sharpe_ratio"])
    test_mdd = float(test_metrics["max_drawdown"])
    test_ann_ret = float(test_metrics["annual_return"])
    test_cum_ret = float(test_metrics["cumulative_return"])
    global_step = best_state["step"] if best_state["step"] >= 0 else 0
    print("=" * 60)
    print(f"[TEST RESULTS @ step {global_step}]")
    print(f"Sharpe: {test_sharpe:.4f}")
    print(f"Annual Return: {test_ann_ret:.4f}")
    print(f"Max Drawdown: {test_mdd:.4f}")
    print(f"Cumulative Return: {test_cum_ret:.4f}")
    print("=" * 60)

    test_metric_dict = {
        "annual_return": test_ann_ret,
        "sharpe_ratio": test_sharpe,
        "max_drawdown": test_mdd,
        "turnover": float(test_metrics["turnover"]),
        "cumulative_return": test_cum_ret,
    }
    test_weights = np.asarray(test_res["weights"])
    T_test = test_weights.shape[0]
    test_dates = dates_ns[split_idx.test][:T_test].astype(np.int64)

    run_logger.log_test_artifacts(
        metrics=test_metric_dict,
        equity_curve=np.asarray(test_res["equity_curve"]),
        portfolio_returns=np.asarray(test_res["portfolio_returns"]),
        weights=test_weights,
        dates=test_dates,
    )

    # Also mirror the canonical spec-required layout at results/test/<run_name>/
    # so downstream analysis tooling can find test outputs at a stable
    # location without inspecting run_dir structure.
    global_test_dir = Path(args.results_dir) / "test" / args.run_name
    global_test_dir.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(global_test_dir / "metrics.csv", "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in sorted(test_metric_dict.keys()):
            writer.writerow([k, test_metric_dict[k]])
    np.save(global_test_dir / "equity_curve.npy", np.asarray(test_res["equity_curve"]))
    np.save(global_test_dir / "portfolio_returns.npy",
            np.asarray(test_res["portfolio_returns"]))
    np.save(global_test_dir / "weights.npy", test_weights)
    np.save(global_test_dir / "dates.npy", test_dates)
    print(f"Test artifacts mirrored to {global_test_dir}")

    run_logger.log_final_metrics({
        "best_val_sharpe": best_state["sharpe"],
        "best_val_step": best_state["step"],
        "test_sharpe": float(test_metrics["sharpe_ratio"]),
        "test_annual_return": float(test_metrics["annual_return"]),
        "test_max_drawdown": float(test_metrics["max_drawdown"]),
        "test_turnover": float(test_metrics["turnover"]),
        "test_cumulative_return": float(test_metrics["cumulative_return"]),
    })
    print(f"[INFO] Training complete. Results saved at: {abs_run_dir}")
    run_logger.close()

    print("Training + evaluation complete.")


if __name__ == "__main__":
    main()
