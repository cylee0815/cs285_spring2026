"""Tests for extended RunLogger behavior.

Covers the new logging/evaluation contract introduced by the audit:

* train-step logs include the extended diagnostics (q_mean, q_std,
  advantage_mean, policy_entropy) when provided,
* every step (train and validation) appends a row to ``logs.jsonl``,
* ``log_validation_curves`` writes canonical artifact files including
  ``dates.npy`` that align with the equity/returns/weights arrays,
* ``log_test_artifacts`` saves the same canonical artifact set
  (weights/returns/equity/dates) in the ``test`` subfolder.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.experiment_logger import RunLogger


def _make_logger(tmp_path: Path) -> RunLogger:
    return RunLogger(
        run_dir=tmp_path,
        run_name="test_run",
        config={"seed": 0},
        wandb_enabled=False,
    )


class TestLogsJsonl:
    def test_train_step_writes_jsonl(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.log_step(1, {
            "v_loss": 0.5, "q_loss": 0.4, "policy_loss": 0.3,
            "q_mean": 1.0, "q_std": 0.1, "advantage_mean": 0.2,
            "policy_entropy": 2.0,
        })
        logger.close()

        path = tmp_path / "logs.jsonl"
        assert path.exists(), "logs.jsonl was not created"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["phase"] == "train"
        assert row["step"] == 1
        for key in ("v_loss", "q_loss", "policy_loss",
                    "q_mean", "q_std", "advantage_mean", "policy_entropy"):
            assert key in row, f"missing {key} in jsonl row"

    def test_validation_metrics_write_jsonl(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        logger.log_validation_metrics(100, {
            "sharpe": 1.2,
            "annual_return": 0.15,
            "max_drawdown": 0.1,
            "turnover": 0.03,
            "cumulative_return": 0.2,
        })
        logger.close()

        rows = [
            json.loads(l)
            for l in (tmp_path / "logs.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1
        assert rows[0]["phase"] == "validation"
        assert rows[0]["step"] == 100
        assert rows[0]["sharpe"] == pytest.approx(1.2)


class TestArtifactPersistence:
    def test_validation_curves_saves_canonical_artifacts(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        T, N = 20, 4
        equity = np.linspace(1.0, 1.2, T)
        rets = np.diff(equity, prepend=1.0) / equity
        weights = np.full((T, N), 1.0 / N)
        dates = np.arange(T, dtype=np.int64) * 86_400_000_000_000  # ns

        logger.log_validation_curves(
            step=500,
            equity_curve=equity,
            portfolio_returns=rets,
            weights=weights,
            dates=dates,
        )
        logger.close()

        val_dir = tmp_path / "validation"
        for fname in ("equity_curve.npy", "portfolio_returns.npy",
                      "weights.npy", "dates.npy"):
            assert (val_dir / fname).exists(), f"{fname} missing from validation dir"

        assert np.load(val_dir / "weights.npy").shape == (T, N)
        assert np.load(val_dir / "dates.npy").shape == (T,)
        assert np.load(val_dir / "portfolio_returns.npy").shape[0] == T

    def test_test_artifacts_saves_canonical_set(self, tmp_path: Path) -> None:
        logger = _make_logger(tmp_path)
        T, N = 15, 3
        equity = np.linspace(1.0, 1.1, T)
        rets = np.diff(equity, prepend=1.0) / equity
        weights = np.full((T, N), 1.0 / N)
        dates = np.arange(T, dtype=np.int64)

        logger.log_test_artifacts(
            metrics={"sharpe_ratio": 1.0, "annual_return": 0.08},
            equity_curve=equity,
            portfolio_returns=rets,
            weights=weights,
            dates=dates,
        )
        logger.close()

        test_dir = tmp_path / "test"
        for fname in ("equity_curve.npy", "portfolio_returns.npy",
                      "weights.npy", "dates.npy", "metrics.csv"):
            assert (test_dir / fname).exists(), f"{fname} missing from test dir"

    def test_dates_alignment_asserted(self, tmp_path: Path) -> None:
        """Mismatched dates length must fail loudly, not silently."""
        logger = _make_logger(tmp_path)
        T = 10
        weights = np.full((T, 4), 0.25)
        equity = np.ones(T)
        rets = np.zeros(T)
        bad_dates = np.arange(T - 1, dtype=np.int64)

        with pytest.raises(AssertionError):
            logger.log_validation_curves(
                step=1,
                equity_curve=equity,
                portfolio_returns=rets,
                weights=weights,
                dates=bad_dates,
            )
        logger.close()
