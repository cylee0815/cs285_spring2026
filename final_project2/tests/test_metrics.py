"""Tests for evaluation/metrics.py — deterministic, no randomness."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import (
    compute_annual_return,
    compute_annual_volatility,
    compute_cumulative_return,
    compute_max_drawdown,
    compute_rolling_sharpe,
    compute_sharpe,
    compute_turnover,
)


# ---------------------------------------------------------------------------
# compute_sharpe
# ---------------------------------------------------------------------------


class TestSharpe:
    def test_constant_positive_returns(self) -> None:
        """Zero vol → Sharpe should be 0.0 (division guard)."""
        returns = np.full(100, 0.001)
        assert compute_sharpe(returns) == 0.0

    def test_known_value(self) -> None:
        """Manually computed Sharpe for a simple returns series."""
        returns = np.array([0.01, -0.005, 0.008, -0.002, 0.006])
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=0)
        expected = mean_r / std_r * np.sqrt(252)
        assert pytest.approx(compute_sharpe(returns), rel=1e-8) == expected

    def test_empty_returns(self) -> None:
        """Empty array → Sharpe 0.0."""
        assert compute_sharpe(np.array([])) == 0.0

    def test_single_return(self) -> None:
        """Single return → zero vol → 0.0."""
        assert compute_sharpe(np.array([0.05])) == 0.0


# ---------------------------------------------------------------------------
# compute_max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_monotonically_increasing(self) -> None:
        """Monotonically increasing equity → MDD == 0."""
        curve = np.linspace(1.0, 2.0, 100)
        assert compute_max_drawdown(curve) == pytest.approx(0.0)

    def test_monotonically_decreasing(self) -> None:
        """Equity goes from 1.0 to 0.5 → MDD == 0.5."""
        curve = np.linspace(1.0, 0.5, 100)
        assert compute_max_drawdown(curve) == pytest.approx(0.5, rel=1e-6)

    def test_known_drawdown(self) -> None:
        """Peak at 2.0, trough at 1.0 → MDD = 0.5."""
        curve = np.array([1.0, 1.5, 2.0, 1.0, 1.5])
        assert compute_max_drawdown(curve) == pytest.approx(0.5, rel=1e-8)

    def test_flat_curve(self) -> None:
        """Flat equity → MDD == 0."""
        curve = np.ones(50)
        assert compute_max_drawdown(curve) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_turnover
# ---------------------------------------------------------------------------


class TestTurnover:
    def test_constant_weights(self) -> None:
        """No rebalancing → turnover == 0."""
        weights = np.tile([0.5, 0.3, 0.2], (20, 1))
        assert compute_turnover(weights) == pytest.approx(0.0)

    def test_known_turnover(self) -> None:
        """Two weights, known L1 distance."""
        weights = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        # L1 distance = |1-0| + |0-1| = 2.0; mean over 1 diff = 2.0
        assert compute_turnover(weights) == pytest.approx(2.0)

    def test_single_row(self) -> None:
        """Single weight row → turnover 0."""
        weights = np.array([[0.5, 0.5]])
        assert compute_turnover(weights) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_cumulative_return
# ---------------------------------------------------------------------------


class TestCumulativeReturn:
    def test_zero_returns(self) -> None:
        """All zero returns → cumulative return == 0."""
        returns = np.zeros(10)
        assert compute_cumulative_return(returns) == pytest.approx(0.0)

    def test_known_cumulative(self) -> None:
        """10% and 20% returns → (1.1)(1.2) - 1 = 0.32."""
        returns = np.array([0.10, 0.20])
        assert compute_cumulative_return(returns) == pytest.approx(0.32, rel=1e-8)

    def test_negative_returns(self) -> None:
        """-50% → cumulative return == -0.5."""
        returns = np.array([-0.5])
        assert compute_cumulative_return(returns) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# compute_annual_return / volatility
# ---------------------------------------------------------------------------


class TestAnnualized:
    def test_annual_return(self) -> None:
        returns = np.array([0.001] * 252)
        expected = 0.001 * 252
        assert compute_annual_return(returns) == pytest.approx(expected, rel=1e-8)

    def test_annual_volatility(self) -> None:
        returns = np.array([0.01, -0.01] * 126)
        daily_std = np.std(returns, ddof=0)
        expected = daily_std * np.sqrt(252)
        assert compute_annual_volatility(returns) == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# compute_rolling_sharpe
# ---------------------------------------------------------------------------


class TestRollingSharpe:
    def test_output_shape(self) -> None:
        returns = np.random.default_rng(0).standard_normal(200) * 0.01
        result = compute_rolling_sharpe(returns, window=63)
        assert result.shape == (200,)

    def test_early_values_nan(self) -> None:
        returns = np.random.default_rng(0).standard_normal(100) * 0.01
        result = compute_rolling_sharpe(returns, window=63)
        assert np.all(np.isnan(result[:62]))
        assert not np.isnan(result[62])

    def test_agrees_with_full_sharpe(self) -> None:
        """Rolling Sharpe on the last window should match compute_sharpe on that slice."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(100) * 0.01
        window = 63
        result = compute_rolling_sharpe(returns, window=window)
        expected = compute_sharpe(returns[-window:])
        assert result[-1] == pytest.approx(expected, rel=1e-8)
