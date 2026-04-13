"""Lightweight experiment logger for ablation runs.

Provides both W&B logging (when available/enabled) and local file logging
for reproducibility and offline plotting.

Local artifacts per run:
    losses.csv               — step, v_loss, q_loss, policy_loss
    logs.jsonl               — one JSON row per train/validation log event
    validation.csv           — one row per validation step
    equity_curve.npy         — test-period equity curve (run_ablation mirror)
    cumulative_return.npy    — test-period cumulative-return curve
    portfolio_returns.npy    — test-period per-step portfolio returns
    weights.npy              — test-period policy actions (T, n_assets)
    metrics.json             — final summary scalars
    config.yaml              — run configuration snapshot
    validation/              — canonical val artifact folder
        equity_curve.npy, portfolio_returns.npy, weights.npy, dates.npy
        latest_*.npy         — per-step history snapshots
    test/                    — canonical test artifact folder
        metrics.csv
        equity_curve.npy, portfolio_returns.npy, weights.npy, dates.npy
"""

from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class _Tee:
    """Duplicate writes to multiple streams (e.g. stdout + a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        for st in self._streams:
            if hasattr(st, "isatty") and st.isatty():
                return True
        return False


# Canonical set of per-step training metrics we log.
_TRAIN_METRIC_KEYS: tuple[str, ...] = (
    "v_loss",
    "q_loss",
    "policy_loss",
    "q_mean",
    "q_std",
    "advantage_mean",
    "policy_entropy",
)


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
        capture_stdout: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or {}
        self._wandb_run = None
        self._train_step = 0  # monotonic step counter for W&B

        # Optional stdout/stderr tee to <run_dir>/stdout.log. Preserve
        # originals so close() can restore them even if wandb isn't used.
        self._stdout_fp = None
        self._orig_stdout = None
        self._orig_stderr = None
        if capture_stdout:
            self._stdout_fp = open(self.run_dir / "stdout.log", "w", buffering=1)
            self._orig_stdout = sys.stdout
            self._orig_stderr = sys.stderr
            sys.stdout = _Tee(self._orig_stdout, self._stdout_fp)
            sys.stderr = _Tee(self._orig_stderr, self._stdout_fp)

        # Moving average buffers (window=100) for losses.
        self._ma_window = 100
        self._ma_buffers: dict[str, collections.deque] = {
            k: collections.deque(maxlen=self._ma_window)
            for k in ("v_loss", "q_loss", "policy_loss")
        }

        # Save config snapshot locally
        if self.config:
            with open(self.run_dir / "config.yaml", "w") as f:
                yaml.safe_dump(self.config, f, sort_keys=False, default_flow_style=False)

        # Initialize losses CSV
        self._losses_path = self.run_dir / "losses.csv"
        self._losses_fp = open(self._losses_path, "w", newline="")
        self._losses_writer = csv.writer(self._losses_fp)
        self._losses_writer.writerow(["step", "v_loss", "q_loss", "policy_loss"])

        # Initialize unified JSONL log (train + val rows, interleaved).
        self._jsonl_path = self.run_dir / "logs.jsonl"
        self._jsonl_fp = open(self._jsonl_path, "w")

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_jsonl(self, row: dict[str, Any]) -> None:
        self._jsonl_fp.write(json.dumps(row, default=_json_default) + "\n")
        # Flush every line so crashes don't lose log state.
        self._jsonl_fp.flush()

    # ------------------------------------------------------------------
    # Train-step logging
    # ------------------------------------------------------------------

    def log_step(self, step: int, metrics: dict[str, float]) -> None:
        """Log per-step training metrics.

        Accepts the full extended metrics dict: ``v_loss``, ``q_loss``,
        ``policy_loss`` plus any diagnostics (``q_mean``, ``q_std``,
        ``advantage_mean``, ``policy_entropy``). Unknown keys are passed
        through to WandB and JSONL unchanged.
        """
        self._train_step = step

        self._losses_writer.writerow([
            step,
            metrics.get("v_loss", ""),
            metrics.get("q_loss", ""),
            metrics.get("policy_loss", ""),
        ])
        if step % 100 == 0:
            self._losses_fp.flush()

        # Update moving-average buffers for losses.
        for key in ("v_loss", "q_loss", "policy_loss"):
            val = metrics.get(key)
            if val is not None:
                self._ma_buffers[key].append(val)

        # JSONL row — always the complete dict, tagged phase=train.
        jsonl_row: dict[str, Any] = {"phase": "train", "step": step}
        for k in _TRAIN_METRIC_KEYS:
            if k in metrics and metrics[k] is not None:
                jsonl_row[k] = float(metrics[k])
        self._append_jsonl(jsonl_row)

        if self._wandb_run is not None:
            import wandb
            log_dict: dict[str, Any] = {"train/step": step}
            for k in _TRAIN_METRIC_KEYS:
                v = metrics.get(k)
                if v is not None:
                    log_dict[f"train/{k}"] = float(v)
            for key in ("v_loss", "q_loss", "policy_loss"):
                buf = self._ma_buffers[key]
                if len(buf) == self._ma_window:
                    log_dict[f"train/{key}_ma100"] = float(np.mean(buf))
            wandb.log(log_dict, step=step)

    # ------------------------------------------------------------------
    # End-of-run evaluation curve logging (ablation convenience wrappers)
    # ------------------------------------------------------------------

    def log_curves(
        self,
        equity_curve: np.ndarray,
        cumulative_return: np.ndarray,
    ) -> None:
        """Save final-eval equity + cumulative-return curves locally and to W&B.

        WandB curves are logged as ``wandb.plot.line`` tables so a single
        artifact is attached rather than a per-step flood.
        """
        equity_curve = np.asarray(equity_curve, dtype=np.float64)
        cumulative_return = np.asarray(cumulative_return, dtype=np.float64)

        np.save(self.run_dir / "equity_curve.npy", equity_curve)
        np.save(self.run_dir / "cumulative_return.npy", cumulative_return)

        if self._wandb_run is not None:
            import wandb
            eq_table = wandb.Table(
                data=[[i, v] for i, v in enumerate(equity_curve.tolist())],
                columns=["t", "equity"],
            )
            cr_table = wandb.Table(
                data=[[i, v] for i, v in enumerate(cumulative_return.tolist())],
                columns=["t", "cumret"],
            )
            wandb.log({
                "eval/equity_curve": wandb.plot.line(
                    eq_table, "t", "equity", title="Equity Curve"
                ),
                "eval/cumulative_return_curve": wandb.plot.line(
                    cr_table, "t", "cumret", title="Cumulative Return"
                ),
            })

    def log_backtest_arrays(
        self,
        portfolio_returns: np.ndarray,
        weights: np.ndarray,
        dates: np.ndarray | None = None,
    ) -> None:
        """Save per-step backtest arrays used by downstream analysis.

        Writes into ``run_dir`` (not the test/ subdir): these are the
        'default' backtest artifacts (currently the test-period backtest
        for an ablation run). The dedicated ``test/`` subdir is populated
        by :meth:`log_test_artifacts`.
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

        np.save(self.run_dir / "portfolio_returns.npy", portfolio_returns)
        np.save(self.run_dir / "weights.npy", weights)
        if dates is not None:
            dates_arr = np.asarray(dates, dtype=np.int64)
            assert dates_arr.shape[0] == weights.shape[0], (
                f"dates length {dates_arr.shape[0]} does not match "
                f"weights length {weights.shape[0]}"
            )
            np.save(self.run_dir / "dates.npy", dates_arr)

    # ------------------------------------------------------------------
    # Validation logging
    # ------------------------------------------------------------------

    def log_validation_metrics(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Log validation-period scalars. Mirrors to ``val/*`` on WandB and
        appends a row to both ``validation.csv`` and ``logs.jsonl``."""
        csv_path = self.run_dir / "validation.csv"
        write_header = not csv_path.exists()
        keys = sorted(metrics.keys())
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", *keys])
            writer.writerow([step, *[metrics[k] for k in keys]])

        jsonl_row: dict[str, Any] = {"phase": "validation", "step": step}
        for k, v in metrics.items():
            if v is not None:
                jsonl_row[k] = float(v)
        self._append_jsonl(jsonl_row)

        if self._wandb_run is not None:
            import wandb
            wb_payload: dict[str, Any] = {f"val/{k}": v for k, v in metrics.items()}
            # Also mirror with the canonical short eval/* keys requested by
            # the project's logging spec so dashboards can query either form.
            _eval_alias = {
                "sharpe_ratio": "eval/sharpe",
                "annual_return": "eval/ann_return",
                "max_drawdown": "eval/mdd",
                "turnover": "eval/turnover",
            }
            for src, dst in _eval_alias.items():
                if src in metrics and metrics[src] is not None:
                    wb_payload[dst] = float(metrics[src])
            wandb.log(wb_payload, step=step)

    def log_validation_curves(
        self,
        step: int,
        equity_curve: np.ndarray,
        portfolio_returns: np.ndarray,
        weights: np.ndarray,
        dates: np.ndarray | None = None,
    ) -> None:
        """Persist the validation equity curve + backtest arrays.

        Writes two layouts:

        * ``run_dir/validation/latest_*.npy`` — per-step-overwriting
          snapshots used by tail-end plotting.
        * ``run_dir/validation/{equity_curve,portfolio_returns,weights,dates}.npy``
          — canonical artifact filenames consumed by ``regime_analysis.py``
          and the tearsheet utility.

        ``dates`` must align with the time dimension of ``equity_curve``,
        ``portfolio_returns``, and ``weights``. A length mismatch is an
        AssertionError rather than a silent truncation.
        """
        val_dir = self.run_dir / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)

        equity_curve = np.asarray(equity_curve)
        portfolio_returns = np.asarray(portfolio_returns)
        weights = np.asarray(weights)

        T = weights.shape[0]
        assert len(portfolio_returns) == T, (
            f"portfolio_returns length {len(portfolio_returns)} != "
            f"weights.shape[0] {T}"
        )
        assert len(equity_curve) == T, (
            f"equity_curve length {len(equity_curve)} != weights.shape[0] {T}"
        )

        np.save(val_dir / "latest_equity_curve.npy", equity_curve)
        np.save(val_dir / "latest_portfolio_returns.npy", portfolio_returns)
        np.save(val_dir / "latest_weights.npy", weights)

        # Canonical filenames so downstream tools don't hunt for a prefix.
        np.save(val_dir / "equity_curve.npy", equity_curve)
        np.save(val_dir / "portfolio_returns.npy", portfolio_returns)
        np.save(val_dir / "weights.npy", weights)

        if dates is not None:
            dates_arr = np.asarray(dates, dtype=np.int64)
            assert dates_arr.shape[0] == T, (
                f"dates length {dates_arr.shape[0]} != weights.shape[0] {T}"
            )
            np.save(val_dir / "dates.npy", dates_arr)
            np.save(val_dir / "latest_dates.npy", dates_arr)

        # W&B line plot (written per-validation call with step suffix).
        if self._wandb_run is not None:
            import wandb
            eq_table = wandb.Table(
                data=[[i, float(v)] for i, v in enumerate(equity_curve)],
                columns=["t", "equity"],
            )
            wandb.log({
                f"val/equity_curve": wandb.plot.line(
                    eq_table, "t", "equity",
                    title=f"Validation Equity Curve @ step {step}",
                ),
            }, step=step)

    # ------------------------------------------------------------------
    # Test logging
    # ------------------------------------------------------------------

    def log_test_artifacts(
        self,
        metrics: dict[str, float],
        equity_curve: np.ndarray,
        portfolio_returns: np.ndarray,
        weights: np.ndarray,
        dates: np.ndarray | None = None,
    ) -> None:
        """Persist the final test-period results (run once post-training).

        Writes the canonical 4-tuple (equity_curve, portfolio_returns,
        weights, dates) plus ``metrics.csv`` into ``run_dir/test/``.
        Mirrors scalar metrics to W&B as ``test/*``.
        """
        test_dir = self.run_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        equity_curve = np.asarray(equity_curve)
        portfolio_returns = np.asarray(portfolio_returns)
        weights = np.asarray(weights)

        T = weights.shape[0]
        assert len(portfolio_returns) == T, (
            f"portfolio_returns length {len(portfolio_returns)} != "
            f"weights.shape[0] {T}"
        )
        assert len(equity_curve) == T, (
            f"equity_curve length {len(equity_curve)} != weights.shape[0] {T}"
        )

        with open(test_dir / "metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k in sorted(metrics.keys()):
                writer.writerow([k, metrics[k]])

        np.save(test_dir / "equity_curve.npy", equity_curve)
        np.save(test_dir / "portfolio_returns.npy", portfolio_returns)
        np.save(test_dir / "weights.npy", weights)
        if dates is not None:
            dates_arr = np.asarray(dates, dtype=np.int64)
            assert dates_arr.shape[0] == T, (
                f"dates length {dates_arr.shape[0]} != weights.shape[0] {T}"
            )
            np.save(test_dir / "dates.npy", dates_arr)

        if self._wandb_run is not None:
            import wandb
            for k, v in metrics.items():
                wandb.run.summary[f"test/{k}"] = v
            # Also attach the test equity curve as a line plot artifact.
            eq_table = wandb.Table(
                data=[[i, float(v)] for i, v in enumerate(equity_curve)],
                columns=["t", "equity"],
            )
            wandb.log({
                "test/equity_curve": wandb.plot.line(
                    eq_table, "t", "equity", title="Test Equity Curve"
                ),
            })

    # ------------------------------------------------------------------
    # Final summary + lifecycle
    # ------------------------------------------------------------------

    def log_final_metrics(self, metrics: dict[str, float]) -> None:
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=_json_default)
        if self._wandb_run is not None:
            import wandb
            for k, v in metrics.items():
                wandb.run.summary[f"final/{k}"] = v

    def make_log_fn(self, log_interval: int = 1000):
        """Return a callback compatible with train_iql's ``log_fn`` parameter."""
        def _log_fn(step: int, metrics: dict[str, float]) -> None:
            self.log_step(step, metrics)
            if step % log_interval == 0:
                v = metrics.get("v_loss", float("nan"))
                q = metrics.get("q_loss", float("nan"))
                p = metrics.get("policy_loss", float("nan"))
                extra = ""
                if "q_mean" in metrics:
                    extra = (
                        f"  q_mean={metrics['q_mean']:+.3f}"
                        f"  adv={metrics.get('advantage_mean', float('nan')):+.3f}"
                        f"  H={metrics.get('policy_entropy', float('nan')):.3f}"
                    )
                print(
                    f"[step {step:>7d}]  v_loss={v:.3e}  "
                    f"q_loss={q:.3e}  policy_loss={p:.4f}{extra}"
                )
        return _log_fn

    def close(self) -> None:
        if not self._losses_fp.closed:
            self._losses_fp.flush()
            self._losses_fp.close()
        if not self._jsonl_fp.closed:
            self._jsonl_fp.flush()
            self._jsonl_fp.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
        # Restore original stdout/stderr if we redirected them.
        if self._stdout_fp is not None:
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr
            try:
                self._stdout_fp.flush()
                self._stdout_fp.close()
            except Exception:
                pass
            self._stdout_fp = None

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for numpy types."""
    if hasattr(o, "item") and callable(o.item):
        try:
            return o.item()
        except (ValueError, TypeError):
            pass
    if hasattr(o, "tolist") and callable(o.tolist):
        return o.tolist()
    return str(o)
