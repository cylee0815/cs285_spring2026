"""Price-data ingestion with a local parquet cache.

Responsibilities
----------------
* Fetch split/dividend-adjusted closing prices for a set of tickers over a
  requested date range.
* Cache each ticker as ``<cache_dir>/<TICKER>.parquet`` so subsequent runs
  do not hit the network (yfinance is rate-limited and occasionally flaky).
* Align tickers onto a common trading-day index (inner-join) so the
  returned frame has no NaNs and can be fed directly to the feature pipeline.
* Be fully testable without network access via dependency injection of the
  ``downloader`` callable.

Usage
-----
As a library::

    from data.download_data import download_prices
    df = download_prices(
        tickers=["SPY", "TLT", "GLD"],
        start="2008-01-01",
        end="2026-03-31",
        cache_dir=".cache/prices",
    )

As a CLI entrypoint, driven by an experiment config::

    uv run python -m data.download_data --config experiments/configs/default.yaml

Pass ``--force`` to bypass the cache and re-download everything.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd

from utils.logging import get_console_logger

__all__ = ["download_prices", "Downloader"]

_logger = get_console_logger("data.download_data")


# Type alias: a downloader takes (tickers, start, end) and returns a DataFrame
# whose columns are the ticker names and whose index is a DatetimeIndex of
# trading days. This signature is intentionally narrower than yfinance's raw
# API so stub downloaders in tests are trivial to write.
Downloader = Callable[[list[str], str, str], pd.DataFrame]


# ---------------------------------------------------------------------------
# Default downloader: wraps yfinance
# ---------------------------------------------------------------------------


def _default_downloader(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices from Yahoo Finance.

    Isolated in its own function so:

    1. the heavy ``yfinance`` import only happens when we actually need the
       network (tests always inject a stub and skip this import entirely),
    2. upgrading yfinance or swapping backends (e.g., Stooq, Tiingo) only
       touches one function, and
    3. the MultiIndex/single-ticker column schema weirdness of yfinance is
       normalized away before returning.
    """
    import yfinance as yf  # local import to keep test boot fast

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,  # applies split + dividend adjustments to OHLC
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {tickers} in [{start}, {end}]. "
            f"Check connectivity, ticker validity, and date range."
        )
    # yfinance returns different shapes depending on len(tickers):
    #   - multi-ticker, group_by='ticker': MultiIndex columns (ticker, field)
    #   - single-ticker: flat columns (field,)
    if isinstance(raw.columns, pd.MultiIndex):
        close = pd.DataFrame(
            {tick: raw[tick]["Close"] for tick in tickers if tick in raw.columns.levels[0]}
        )
    else:
        # Single-ticker path — the column "Close" holds the adjusted close.
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index)
    return close


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str | Path,
    *,
    use_cache: bool = True,
    downloader: Downloader | None = None,
) -> pd.DataFrame:
    """Return a clean ``(T, N_tickers)`` adjusted-close price frame.

    Parameters
    ----------
    tickers:
        List of ticker symbols (e.g., ``["SPY", "TLT", "GLD"]``). The
        returned frame preserves this exact column order.
    start, end:
        Inclusive ISO date strings (``YYYY-MM-DD``).
    cache_dir:
        Directory to store per-ticker parquet files. Created on demand.
    use_cache:
        If True (default), tickers whose parquet file already exists are
        read from disk and not re-downloaded. Set to False to force a
        refresh (e.g., after Yahoo backfills a recent day).
    downloader:
        Optional ``(tickers, start, end) -> DataFrame`` callable used in
        place of the default yfinance backend. Tests pass a stub here so
        nothing ever touches the network.

    Returns
    -------
    pandas.DataFrame
        Indexed by DatetimeIndex (business days), columns in the exact
        order of ``tickers``, guaranteed free of NaN values (via
        inner-join on common trading days), clipped to ``[start, end]``.

    Raises
    ------
    ValueError
        If the downloader fails to return data for a requested ticker, or
        if a cached file is empty.
    RuntimeError
        If the requested window has no common trading days across all
        tickers (e.g., date range is entirely before a ticker's inception).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Partition tickers into "already cached" vs "needs downloading".
    # With use_cache=False the cached partition is empty (force refresh).
    to_download: list[str] = []
    for tick in tickers:
        cached_path = cache_dir / f"{tick}.parquet"
        if use_cache and cached_path.exists():
            continue
        to_download.append(tick)

    # Fetch missing tickers in a single downloader call (yfinance is more
    # efficient as a batch than per-ticker loops).
    if to_download:
        dl = downloader or _default_downloader
        _logger.info(
            "Downloading %d ticker(s): %s from %s to %s",
            len(to_download), to_download, start, end,
        )
        fetched = dl(to_download, start, end)
        if not isinstance(fetched, pd.DataFrame):
            raise TypeError(
                f"downloader must return a DataFrame; got {type(fetched).__name__}"
            )
        for tick in to_download:
            if tick not in fetched.columns:
                raise ValueError(
                    f"Downloader did not return data for ticker {tick!r}. "
                    f"Check the symbol or the downloader's date range."
                )
            series = fetched[tick].dropna()
            if series.empty:
                # A valid ticker with no data in the requested window is
                # NOT a hard error here — the assembly step below will raise
                # a single, uniform "no common trading days" RuntimeError.
                # Skipping the write also avoids poisoning the cache with
                # empty parquet files that would suppress future retries.
                _logger.warning(
                    "Downloader returned empty data for %r in [%s, %s]; "
                    "will raise downstream once the combined frame is checked.",
                    tick, start, end,
                )
                continue
            # Persist as a single-column DataFrame so the parquet schema is
            # stable regardless of whether the source gave us a Series.
            series.to_frame(name=tick).to_parquet(cache_dir / f"{tick}.parquet")

    # Load every ticker from cache into a dict of Series so we can later
    # assemble them in the requested column order. A missing cache file is
    # treated as an empty series — this happens when the downloader
    # returned no data for that ticker in the requested window (we skip
    # caching empty frames). The downstream inner-join will collapse the
    # whole thing to empty and raise a uniform RuntimeError.
    series_map: dict[str, pd.Series] = {}
    for tick in tickers:
        path = cache_dir / f"{tick}.parquet"
        if not path.exists():
            series_map[tick] = pd.Series(
                dtype=float, name=tick, index=pd.DatetimeIndex([], name="date")
            )
            continue
        df = pd.read_parquet(path)
        if tick not in df.columns:
            raise ValueError(
                f"Cache file {path} is missing column {tick!r}; "
                f"delete it and re-download."
            )
        s = df[tick]
        s.index = pd.to_datetime(s.index)
        series_map[tick] = s

    # Assemble into a wide frame, preserving the caller's ticker order.
    # Outer-join first so every date that any ticker has is represented, then
    # drop NaNs to enforce a common trading-day calendar.
    combined = pd.concat(
        [series_map[t].rename(t) for t in tickers], axis=1, join="outer"
    ).sort_index()

    # Clip to [start, end] BEFORE inner-joining, so out-of-range rows from
    # older cache files don't poison the window.
    window_mask = (
        (combined.index >= pd.Timestamp(start))
        & (combined.index <= pd.Timestamp(end))
    )
    combined = combined.loc[window_mask]
    combined = combined.dropna(how="any")

    if combined.empty:
        raise RuntimeError(
            f"No common trading days across {tickers} in [{start}, {end}]. "
            f"At least one ticker has no data in this window."
        )

    # Friendly warning: the first available date may be much later than
    # requested (e.g., UUP did not trade before 2007-02), which silently
    # shrinks the train window. Surface it so users notice.
    first_date = combined.index.min()
    if (first_date - pd.Timestamp(start)).days > 30:
        _logger.warning(
            "First common trading day %s is >30 days after requested start "
            "%s. At least one ticker has limited history in this window.",
            first_date.date(), start,
        )

    combined.index.name = "date"
    return combined[tickers]  # ensure exact requested column order


# ---------------------------------------------------------------------------
# CLI entrypoint: `uv run python -m data.download_data --config ...`
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data.download_data",
        description="Download and cache adjusted close prices for an experiment.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment YAML config file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the local cache and re-download every ticker.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: load a config and download the data it references."""
    # Local imports to keep library-mode imports lightweight.
    from utils.config import load_config

    args = _build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    df = download_prices(
        tickers=list(cfg.data.tickers),
        start=cfg.data.start,
        end=cfg.data.end,
        cache_dir=cfg.data.cache_dir,
        use_cache=not args.force,
    )
    _logger.info(
        "Downloaded %d rows × %d tickers (%s → %s)",
        len(df), df.shape[1], df.index.min().date(), df.index.max().date(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
