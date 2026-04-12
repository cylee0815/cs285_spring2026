"""Tests for ``data.download_data.download_prices``.

We never hit the network in tests: the downloader accepts a caller-supplied
``downloader`` function, and these tests pass in stubs that return
deterministic fake price data. This lets us exercise:

* cache write on first call
* cache read on subsequent calls (stub must not be re-invoked)
* date-range filtering
* inner-join on common trading days across tickers
* missing-ticker error handling
* empty-data error handling
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.download_data import download_prices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_prices(
    tickers: list[str],
    start: str,
    end: str,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic geometric-random-walk prices for testing."""
    dates = pd.date_range(start, end, freq="B")
    rng = np.random.default_rng(seed)
    out = {}
    for i, tick in enumerate(tickers):
        log_rets = rng.normal(loc=0.0003, scale=0.01, size=len(dates))
        prices = 100.0 * np.exp(np.cumsum(log_rets))
        out[tick] = prices + i  # small offset so tickers are distinguishable
    df = pd.DataFrame(out, index=dates)
    df.index.name = "Date"
    return df


class StubDownloader:
    """Records call count so tests can assert cache behavior."""

    def __init__(self, fake_frame: pd.DataFrame) -> None:
        self.fake_frame = fake_frame
        self.calls: list[tuple[tuple[str, ...], str, str]] = []

    def __call__(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        self.calls.append((tuple(tickers), start, end))
        # Return only columns the stub actually knows about, so the unit
        # under test sees the "missing ticker" case the same way it would
        # see yfinance silently dropping a bad symbol.
        available = [t for t in tickers if t in self.fake_frame.columns]
        mask = (self.fake_frame.index >= pd.Timestamp(start)) & (
            self.fake_frame.index <= pd.Timestamp(end)
        )
        sub = self.fake_frame.loc[mask, available]
        return sub.copy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_download_writes_cache_on_first_call(tmp_path: Path) -> None:
    tickers = ["SPY", "TLT"]
    fake = _make_fake_prices(tickers, "2020-01-01", "2020-12-31")
    stub = StubDownloader(fake)

    df = download_prices(
        tickers=tickers,
        start="2020-01-01",
        end="2020-12-31",
        cache_dir=tmp_path,
        downloader=stub,
    )
    # One network call, two cache files written, one per ticker.
    assert len(stub.calls) == 1
    for tick in tickers:
        assert (tmp_path / f"{tick}.parquet").exists()
    # Return value should match the fake data exactly.
    assert list(df.columns) == tickers
    assert len(df) > 0


def test_second_call_hits_cache(tmp_path: Path) -> None:
    tickers = ["SPY", "TLT"]
    fake = _make_fake_prices(tickers, "2020-01-01", "2020-12-31")
    stub = StubDownloader(fake)

    # First call — downloader invoked.
    download_prices(
        tickers=tickers,
        start="2020-01-01",
        end="2020-12-31",
        cache_dir=tmp_path,
        downloader=stub,
    )
    assert len(stub.calls) == 1

    # Second call — everything is cached, so the downloader must NOT be
    # called. We pass a different stub that would crash loudly if invoked.
    def exploding(_tickers, _start, _end):  # pragma: no cover - must not run
        raise AssertionError("Cache should have prevented this call.")

    df2 = download_prices(
        tickers=tickers,
        start="2020-01-01",
        end="2020-12-31",
        cache_dir=tmp_path,
        downloader=exploding,
    )
    assert list(df2.columns) == tickers


def test_use_cache_false_forces_redownload(tmp_path: Path) -> None:
    tickers = ["SPY"]
    fake = _make_fake_prices(tickers, "2020-01-01", "2020-12-31")
    stub = StubDownloader(fake)

    download_prices(
        tickers=tickers,
        start="2020-01-01",
        end="2020-12-31",
        cache_dir=tmp_path,
        downloader=stub,
    )
    assert len(stub.calls) == 1

    download_prices(
        tickers=tickers,
        start="2020-01-01",
        end="2020-12-31",
        cache_dir=tmp_path,
        downloader=stub,
        use_cache=False,
    )
    # With use_cache=False the downloader is called again.
    assert len(stub.calls) == 2


def test_date_range_filter_applied_to_returned_frame(tmp_path: Path) -> None:
    """Even if the cache covers a wider range, returned frame is clipped."""
    tickers = ["SPY"]
    fake = _make_fake_prices(tickers, "2018-01-01", "2022-12-31")
    stub = StubDownloader(fake)

    df = download_prices(
        tickers=tickers,
        start="2020-06-01",
        end="2020-07-31",
        cache_dir=tmp_path,
        downloader=stub,
    )
    assert df.index.min() >= pd.Timestamp("2020-06-01")
    assert df.index.max() <= pd.Timestamp("2020-07-31")
    assert not df.empty


def test_inner_join_drops_rows_missing_any_ticker(tmp_path: Path) -> None:
    """A date present in one ticker but not another must be dropped."""
    tickers = ["SPY", "TLT"]
    dates_full = pd.date_range("2020-01-01", "2020-01-31", freq="B")

    spy = pd.DataFrame({"SPY": np.arange(len(dates_full), dtype=float) + 100.0},
                       index=dates_full)
    # TLT is missing the first 3 business days.
    tlt_dates = dates_full[3:]
    tlt = pd.DataFrame({"TLT": np.arange(len(tlt_dates), dtype=float) + 50.0},
                       index=tlt_dates)

    # Wide dataframe with NaNs where TLT is absent
    fake = spy.join(tlt, how="outer")

    stub = StubDownloader(fake)
    df = download_prices(
        tickers=tickers,
        start="2020-01-01",
        end="2020-01-31",
        cache_dir=tmp_path,
        downloader=stub,
    )
    # The returned frame must contain no NaNs and must start on the first
    # day both tickers were present.
    assert not df.isna().any().any()
    assert df.index.min() == tlt_dates[0]


def test_missing_ticker_from_downloader_raises(tmp_path: Path) -> None:
    tickers = ["SPY", "NONEXISTENT"]
    fake = _make_fake_prices(["SPY"], "2020-01-01", "2020-12-31")
    # Stub only knows about SPY.
    stub = StubDownloader(fake)
    with pytest.raises(ValueError, match="NONEXISTENT"):
        download_prices(
            tickers=tickers,
            start="2020-01-01",
            end="2020-12-31",
            cache_dir=tmp_path,
            downloader=stub,
        )


def test_empty_intersection_raises(tmp_path: Path) -> None:
    """Requesting a window outside the cached data is an error, not silent empty."""
    tickers = ["SPY"]
    fake = _make_fake_prices(tickers, "2020-01-01", "2020-12-31")
    stub = StubDownloader(fake)
    with pytest.raises(RuntimeError, match="No common trading days"):
        download_prices(
            tickers=tickers,
            start="2030-01-01",
            end="2030-12-31",
            cache_dir=tmp_path,
            downloader=stub,
        )


def test_preserves_ticker_column_order(tmp_path: Path) -> None:
    tickers = ["TLT", "SPY", "GLD"]
    fake = _make_fake_prices(tickers, "2020-01-01", "2020-12-31")
    stub = StubDownloader(fake)
    df = download_prices(
        tickers=tickers,
        start="2020-01-01",
        end="2020-06-30",
        cache_dir=tmp_path,
        downloader=stub,
    )
    assert list(df.columns) == tickers


def test_cache_files_are_valid_parquet_roundtrip(tmp_path: Path) -> None:
    """Hand-read the cache files with pandas to confirm the on-disk format."""
    tickers = ["SPY"]
    fake = _make_fake_prices(tickers, "2020-01-01", "2020-06-30")
    stub = StubDownloader(fake)
    download_prices(
        tickers=tickers,
        start="2020-01-01",
        end="2020-06-30",
        cache_dir=tmp_path,
        downloader=stub,
    )
    cached = pd.read_parquet(tmp_path / "SPY.parquet")
    assert "SPY" in cached.columns
    assert len(cached) > 0
