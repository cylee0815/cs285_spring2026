"""Tests for date-based train/val/test split logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.splits import (
    DEFAULT_SPLIT,
    SplitConfig,
    assert_disjoint,
    compute_split_indices,
)


def _to_ns(dates: pd.DatetimeIndex) -> np.ndarray:
    return dates.values.astype("datetime64[ns]").astype(np.int64)


def test_split_config_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        SplitConfig(
            train_start="2008-01-01",
            train_end="2021-06-30",  # overlaps val
            val_start="2021-01-01",
            val_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2026-03-31",
        )


def test_compute_split_indices_is_disjoint_and_time_ordered() -> None:
    dates = pd.bdate_range(start="2008-01-02", end="2026-03-30")
    idx = compute_split_indices(_to_ns(dates), DEFAULT_SPLIT)
    assert idx.train.size > 0
    assert idx.val.size > 0
    assert idx.test.size > 0

    # Pairwise disjoint.
    assert_disjoint(idx.train, idx.val, idx.test)

    # Time ordered.
    assert dates[idx.train[-1]] <= pd.Timestamp(DEFAULT_SPLIT.train_end)
    assert dates[idx.val[0]] >= pd.Timestamp(DEFAULT_SPLIT.val_start)
    assert dates[idx.val[-1]] <= pd.Timestamp(DEFAULT_SPLIT.val_end)
    assert dates[idx.test[0]] >= pd.Timestamp(DEFAULT_SPLIT.test_start)
    assert dates[idx.train[-1]] < dates[idx.val[0]]
    assert dates[idx.val[-1]] < dates[idx.test[0]]


def test_compute_split_indices_accepts_datetimeindex() -> None:
    dates = pd.bdate_range(start="2020-06-01", end="2023-06-01")
    idx = compute_split_indices(dates, DEFAULT_SPLIT)
    # Train portion: 2020-06-01 .. 2020-12-31 inclusive, val 2021, test 2022..
    assert idx.train.size > 0 and idx.val.size > 0 and idx.test.size > 0


def test_assert_disjoint_catches_overlap() -> None:
    a = np.array([0, 1, 2], dtype=np.int64)
    b = np.array([2, 3], dtype=np.int64)
    with pytest.raises(AssertionError):
        assert_disjoint(a, b)


def test_empty_split_raises() -> None:
    dates = pd.bdate_range(start="2024-01-01", end="2024-12-31")
    with pytest.raises(ValueError, match="empty"):
        compute_split_indices(_to_ns(dates), DEFAULT_SPLIT)
