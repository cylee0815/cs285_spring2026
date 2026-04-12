"""Smoke test for the tearsheet utility."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis.plot_training_curves import build_tearsheet


def test_build_tearsheet_creates_png(tmp_path: Path) -> None:
    T_train, T_val, T_test = 100, 30, 50
    rng = np.random.default_rng(0)

    def _equity(T):
        rets = rng.normal(0.0005, 0.01, T)
        return np.cumprod(1 + rets), rets

    eq_train, rets_train = _equity(T_train)
    eq_val, rets_val = _equity(T_val)
    eq_test, rets_test = _equity(T_test)

    # Fake date arrays — just sequential trading days.
    def _dates(n, start):
        import pandas as pd
        return pd.bdate_range(
            start=start, periods=n
        ).values.astype("datetime64[ns]").astype(np.int64)

    out = build_tearsheet(
        output_path=tmp_path / "figures" / "tearsheet.png",
        train={"equity": eq_train, "returns": rets_train,
               "dates": _dates(T_train, "2010-01-04")},
        val={"equity": eq_val, "returns": rets_val,
             "dates": _dates(T_val, "2021-01-04")},
        test={"equity": eq_test, "returns": rets_test,
              "dates": _dates(T_test, "2022-01-03")},
    )
    assert out.exists()
    assert out.stat().st_size > 0
