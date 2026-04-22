"""
Macroeconomic and fundamental feature pipeline for portfolio RL.

Global macro features per date (n_macro = 8), appended to per-asset features in PortfolioEnv:
  0: dgs10        — 10-Year Treasury yield (% p.a.)
  1: dff          — Effective Fed Funds Rate (% p.a.)
  2: yield_spread — DGS10 − DFF (term premium / risk-appetite proxy)
  3: cpi_yoy      — CPI year-over-year % change (inflation regime)
  4: unrate       — Unemployment rate (%)
  5: gdp_growth   — Real GDP QoQ growth, annualized (%)
  6: umcsent_norm — U of M Consumer Sentiment, normalized to [−1, 1]
  7: nfci         — Chicago Fed National Financial Conditions Index

Why these features:
  - yield_spread collapses during recession / flight-to-quality (identifies GFC, COVID)
  - cpi_yoy captures the 2022 inflation shock that broke stock/bond correlation
  - nfci directly encodes financial stress — high NFCI = tight conditions (regime signal)
  - umcsent is a forward-looking sentiment indicator available since 1978

Fundamental data (included in macro features as market-level valuation signals):
  - yield_spread serves as the equity risk premium proxy (no free ETF P/E history)
  - DGS10 level encodes the discount rate used to value all long-duration assets

Requires a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
Set env var FRED_API_KEY or pass api_key= argument.
If unavailable, all macro features are returned as zeros so the pipeline still works.
"""
import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple

N_MACRO_FEATURES = 8

# FRED series IDs and descriptions
_FRED_SERIES = {
    "DGS10":    "10-Year Treasury Constant Maturity Rate",
    "DFF":      "Effective Federal Funds Rate",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers",
    "UNRATE":   "Unemployment Rate",
    "GDPC1":    "Real Gross Domestic Product",
    "UMCSENT":  "University of Michigan: Consumer Sentiment",
    "NFCI":     "Chicago Fed National Financial Conditions Index",
}


def download_fred_macro(
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download macroeconomic series from FRED and align to daily trading dates.

    Returns a DataFrame with columns:
      dgs10, dff, yield_spread, cpi_yoy, unrate, gdp_growth, umcsent_norm, nfci
    indexed by business-day dates in [start, end], forward-filled.

    If api_key is None, reads from FRED_API_KEY env var.
    If the key is missing or the download fails, returns a zero-filled DataFrame.
    """
    api_key = api_key or os.environ.get("FRED_API_KEY", "")
    if not api_key:
        warnings.warn(
            "FRED_API_KEY not set — macro features will be all zeros. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set it with: export FRED_API_KEY=your_key",
            UserWarning,
            stacklevel=2,
        )
        return _zero_macro_df(start, end)

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
    except ImportError:
        warnings.warn("fredapi not installed. Run: uv add fredapi", UserWarning)
        return _zero_macro_df(start, end)

    try:
        raw = {}
        for series_id in _FRED_SERIES:
            try:
                raw[series_id] = fred.get_series(series_id, observation_start=start, observation_end=end)
            except Exception as e:
                warnings.warn(f"FRED series {series_id} failed: {e}")
                raw[series_id] = pd.Series(dtype=float)

        # Build a daily business-day index
        bdays = pd.bdate_range(start=start, end=end)
        df = pd.DataFrame(index=bdays)

        def _align(series: pd.Series, fillna_val: float = 0.0) -> pd.Series:
            """Reindex to daily, forward-fill, then fill any remaining NaNs."""
            if series.empty:
                return pd.Series(fillna_val, index=df.index)
            s = series.reindex(df.index, method="ffill")
            s = s.ffill().bfill().fillna(fillna_val)
            return s

        df["dgs10"]  = _align(raw["DGS10"])
        df["dff"]    = _align(raw["DFF"])
        df["yield_spread"] = df["dgs10"] - df["dff"]

        # CPI: year-over-year % change
        cpi = raw["CPIAUCSL"].resample("D").interpolate()
        cpi_yoy = cpi.pct_change(periods=252).multiply(100)  # ~252 trading days in a year
        df["cpi_yoy"] = _align(cpi_yoy)

        df["unrate"] = _align(raw["UNRATE"])

        # GDP: QoQ annualized growth rate
        gdp = raw["GDPC1"]
        gdp_qoq = gdp.pct_change().multiply(400)  # annualized: × 4, then × 100
        df["gdp_growth"] = _align(gdp_qoq, fillna_val=2.0)  # ~2% long-run default

        # Consumer sentiment: normalize to [−1, 1] (historical range ~50–110)
        umcsent = raw["UMCSENT"]
        umcsent_norm = (umcsent - 80.0) / 30.0  # center on 80, scale by ~1 std
        df["umcsent_norm"] = _align(umcsent_norm)

        df["nfci"] = _align(raw["NFCI"])

        print(f"  [Macro] Downloaded {N_MACRO_FEATURES} FRED features: "
              f"{df.index[0].date()} → {df.index[-1].date()} ({len(df)} days)")
        return df.astype(np.float32)

    except Exception as e:
        warnings.warn(f"FRED download failed ({e}), using zero macro features.", UserWarning)
        return _zero_macro_df(start, end)


def _zero_macro_df(start: str, end: str) -> pd.DataFrame:
    bdays = pd.bdate_range(start=start, end=end)
    cols = ["dgs10", "dff", "yield_spread", "cpi_yoy", "unrate", "gdp_growth", "umcsent_norm", "nfci"]
    return pd.DataFrame(0.0, index=bdays, columns=cols, dtype=np.float32)


def align_macro_to_prices(
    macro_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Align macro DataFrame to the price time index.

    Returns:
        array of shape (T, N_MACRO_FEATURES), float32, forward-filled and normalized.
    """
    aligned = macro_df.reindex(price_index, method="ffill").ffill().bfill().fillna(0.0)
    arr = aligned.values.astype(np.float32)
    # Soft clip to ±5 to avoid extreme outliers dominating the observation
    arr = np.clip(arr, -5.0, 5.0)
    return arr
