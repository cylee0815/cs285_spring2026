"""Lightweight experiment logger for ablation runs.

Provides both W&B logging (when available/enabled) and local file logging
for reproducibility and offline plotting.

Local artifacts per run:
    losses.csv          — step,v_loss,q_loss,policy_loss
    equity_curve.npy    — (T,) equity curve
    cumulative_return.npy — (T,) cumulative return curve
    metrics.json        — final summary scalars
    config.yaml         — run configuration snapshot
"""

from __future__ import annotations

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
        self._losses_writer.writerow([
            step,
            metrics.get("v_loss", ""),
            metrics.get("q_loss", ""),
            metrics.get("policy_loss", ""),
        ])
        # Flush periodically (every 100 steps) to avoid data loss
        if step % 100 == 0:
            self._losses_fp.flush()

        if self._wandb_run is not None:
            import wandb
            wandb.log({
                "v_loss": metrics.get("v_loss"),
                "q_loss": metrics.get("q_loss"),
                "policy_loss": metrics.get("policy_loss"),
                "step": step,
            }, step=step)

    def log_curves(
        self,
        equity_curve: np.ndarray,
        cumulative_return: np.ndarray,
    ) -> None:
        """Save full time-series curves locally and to W&B."""
        equity_curve = np.asarray(equity_curve, dtype=np.float64)
        cumulative_return = np.asarray(cumulative_return, dtype=np.float64)

        np.save(self.run_dir / "equity_curve.npy", equity_curve)
        np.save(self.run_dir / "cumulative_return.npy", cumulative_return)

        if self._wandb_run is not None:
            import wandb
            for t in range(len(equity_curve)):
                wandb.log({
                    "eval/equity_curve": equity_curve[t],
                    "eval/cumulative_return": cumulative_return[t],
                    "eval_step": t,
                }, step=t)

    def log_final_metrics(self, metrics: dict[str, float]) -> None:
        """Save final summary metrics as JSON and to W&B summary."""
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)

        if self._wandb_run is not None:
            import wandb
            for k, v in metrics.items():
                wandb.run.summary[k] = v

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
