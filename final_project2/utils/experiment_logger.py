"""Lightweight experiment logger for ablation runs.

Provides both W&B logging (when available/enabled) and local file logging
for reproducibility and offline plotting.

Local artifacts per run:
    losses.csv          — step,v_loss,q_loss,policy_loss
    equity_curve.npy    — (T,) equity curve
    cumulative_return.npy — (T,) cumulative return curve
    portfolio_returns.npy — (T,) per-step portfolio returns used during backtest
    weights.npy         — (T, n_assets) policy actions taken during backtest
    metrics.json        — final summary scalars
    config.yaml         — run configuration snapshot
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _try_import_wandb():
    """Import wandb if available, return None otherwise."""
    try:
        import wandb
        return wandb
    except ImportError:
        return None


class RunLogger:
    """Combined W&B + local file logger for a single experiment run.

    Parameters
    ----------
    run_dir:
        Local directory for artifacts. Created if it doesn't exist.
    run_name:
        Human-readable run name (used for W&B).
    config:
        Hyperparameters dict — logged to W&B and saved as config.yaml.
    wandb_enabled:
        If True and wandb is installed, initializes a W&B run.
    wandb_project:
        W&B project name.
    """

    def __init__(
        self,
        run_dir: str | Path,
        run_name: str = "",
        config: dict[str, Any] | None = None,
        wandb_enabled: bool = False,
        wandb_project: str = "offline-rl-portfolio",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._wandb_run = None
        self._train_step = 0  # monotonic step counter for W&B

        # Moving average buffers (window=100)
        self._ma_window = 100
        self._ma_buffers: dict[str, collections.deque] = {
            "v_loss": collections.deque(maxlen=100),
            "q_loss": collections.deque(maxlen=100),
            "policy_loss": collections.deque(maxlen=100),
        }

        # Save config snapshot locally
        if self.config:
            with open(self.run_dir / "config.yaml", "w") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)

        # Initialize losses CSV
        self._losses_path = self.run_dir / "losses.csv"
        self._losses_fp = open(self._losses_path, "w", newline="")
        self._losses_writer = csv.writer(self._losses_fp)
        self._losses_writer.writerow(["step", "v_loss", "q_loss", "policy_loss"])

        # W&B init
        if wandb_enabled:
            wandb = _try_import_wandb()
            if wandb is not None:
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_name or None,
                    config=self.config,
                    reinit=True,
                )

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        """Log per-step training losses."""
        self._train_step = step

        self._losses_writer.writerow([
            step,
            metrics.get("v_loss", ""),
            metrics.get("q_loss", ""),
            metrics.get("policy_loss", ""),
        ])
        # Flush periodically (every 100 steps) to avoid data loss
        if step % 100 == 0:
            self._losses_fp.flush()

        # Update moving average buffers
        for key in ("v_loss", "q_loss", "policy_loss"):
            val = metrics.get(key)
            if val is not None:
                self._ma_buffers[key].append(val)

        if self._wandb_run is not None:
            import wandb
            log_dict = {
                "train/v_loss": metrics.get("v_loss"),
                "train/q_loss": metrics.get("q_loss"),
                "train/policy_loss": metrics.get("policy_loss"),
                "train/step": step,
            }
            # Add moving averages once buffer is full
            for key in ("v_loss", "q_loss", "policy_loss"):
                buf = self._ma_buffers[key]
                if len(buf) == self._ma_window:
                    log_dict[f"train/{key}_ma100"] = float(np.mean(buf))
            wandb.log(log_dict, step=step)

    def log_curves(
        self,
        equity_curve: np.ndarray,
        cumulative_return: np.ndarray,
    ) -> None:
        """Save full time-series curves locally and to W&B.

        Eval curves are logged using a custom ``eval_step`` x-axis to avoid
        colliding with the training ``step`` counter.
        """
        equity_curve = np.asarray(equity_curve, dtype=np.float64)
        cumulative_return = np.asarray(cumulative_return, dtype=np.float64)

        np.save(self.run_dir / "equity_curve.npy", equity_curve)
        np.save(self.run_dir / "cumulative_return.npy", cumulative_return)

        if self._wandb_run is not None:
            import wandb

            # Define a custom x-axis so eval curves don't clobber training logs
            wandb.define_metric("eval_step")
            wandb.define_metric("eval/*", step_metric="eval_step")

            n_logged = 0
            for t in range(len(equity_curve)):
                wandb.log({
                    "eval/equity_curve": equity_curve[t],
                    "eval/cumulative_return": cumulative_return[t],
                    "eval_step": t,
                })
                n_logged += 1
            print(f"  WANDB LOG: logged {n_logged} eval curve points "
                  f"(equity_curve, cumulative_return)")

    def log_backtest_arrays(
        self,
        portfolio_returns: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Save per-step backtest arrays used by downstream analysis.

        Saves two files into ``run_dir``:

        * ``portfolio_returns.npy`` — shape ``(T,)``
        * ``weights.npy``           — shape ``(T, n_assets)``

        These are the *actual* actions taken (and resulting returns) during
        evaluation; they must not be recomputed afterwards. The arrays are
        validated to have matching time dimensions before writing.
        """
        portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        assert weights.ndim == 2, (
            f"weights must be 2-D (T, n_assets); got shape {weights.shape}"
        )
        assert weights.shape[0] == len(portfolio_returns), (
            f"weights.shape[0]={weights.shape[0]} does not match "
            f"len(portfolio_returns)={len(portfolio_returns)}"
        )

        returns_path = self.run_dir / "portfolio_returns.npy"
        weights_path = self.run_dir / "weights.npy"
        np.save(returns_path, portfolio_returns)
        np.save(weights_path, weights)
        print(f"  Saved portfolio_returns.npy ({returns_path})")
        print(f"  Saved weights.npy ({weights_path})")

    def log_validation_metrics(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Log validation-period metrics at a given training step.

        Appends a row to ``validation.csv`` and mirrors the scalars to W&B
        under the ``val/`` namespace so the training and validation curves
        can be overlaid on a shared step axis.
        """
        csv_path = self.run_dir / "validation.csv"
        write_header = not csv_path.exists()
        keys = sorted(metrics.keys())
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", *keys])
            writer.writerow([step, *[metrics[k] for k in keys]])

        if self._wandb_run is not None:
            import wandb
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)

    def log_validation_curves(
        self,
        step: int,
        equity_curve: np.ndarray,
        portfolio_returns: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Persist the validation equity curve (and backtest arrays) for a
        given training step. Always overwrites the ``latest_*`` files so
        downstream plotting sees the most recent val run."""
        val_dir = self.run_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)

        np.save(val_dir / "latest_equity_curve.npy", np.asarray(equity_curve))
        np.save(val_dir / "latest_portfolio_returns.npy", np.asarray(portfolio_returns))
        np.save(val_dir / "latest_weights.npy", np.asarray(weights))

    def log_test_artifacts(
        self,
        metrics: dict[str, float],
        equity_curve: np.ndarray,
        portfolio_returns: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Persist the final test-period results (run once post-training).

        Writes:

        * ``results/test/metrics.csv`` (relative to the run dir's parent
          layout; we place it under ``run_dir / "test"``),
        * ``equity_curve.npy``, ``portfolio_returns.npy``, ``weights.npy``.

        Mirrors the scalar metrics to W&B as ``test/*``.
        """
        test_dir = self.run_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        # CSV with one metric per row for easy downstream parsing.
        with open(test_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k in sorted(metrics.keys()):
                writer.writerow([k, metrics[k]])

        np.save(test_dir / "equity_curve.npy", np.asarray(equity_curve))
        np.save(test_dir / "portfolio_returns.npy", np.asarray(portfolio_returns))
        np.save(test_dir / "weights.npy", np.asarray(weights))

        if self._wandb_run is not None:
            import wandb
            for k, v in metrics.items():
                wandb.run.summary[f"test/{k}"] = v

    def log_final_metrics(self, metrics: dict[str, float]) -> None:
        """Save final summary metrics as JSON and to W&B summary."""
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)

        if self._wandb_run is not None:
            import wandb
            for k, v in metrics.items():
                wandb.run.summary[f"final/{k}"] = v
            print(f"  WANDB LOG: final metrics keys={list(metrics.keys())}")

    def make_log_fn(self, log_interval: int = 1000):
        """Return a callback compatible with train_iql's log_fn parameter."""
        def _log_fn(step: int, metrics: dict[str, float]) -> None:
            self.log_step(step, metrics)
            if step % log_interval == 0:
                v = metrics["v_loss"]
                q = metrics["q_loss"]
                p = metrics["policy_loss"]
                print(f"[step {step:>7d}]  v_loss={v:.4f}  q_loss={q:.4f}  policy_loss={p:.4f}")
        return _log_fn

    def close(self) -> None:
        """Flush and close all file handles and W&B run."""
        if not self._losses_fp.closed:
            self._losses_fp.flush()
            self._losses_fp.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for numpy types."""
    if hasattr(o, "item") and callable(o.item):
        return o.item()
    if hasattr(o, "tolist") and callable(o.tolist):
        return o.tolist()
    return str(o)
