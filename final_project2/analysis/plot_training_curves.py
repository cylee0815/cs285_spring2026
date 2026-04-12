"""Training / validation / test tearsheet plot.

Produces a multi-panel PNG summarizing a single run:

* Equity curve with train/val/test regions shaded.
* Drawdown curve (from the equity curve).
* Cumulative return curve.

Expected input layout (produced by the updated ``RunLogger``):

    <run_dir>/
        validation/
            equity_curve.npy, portfolio_returns.npy, weights.npy, dates.npy
        test/
            equity_curve.npy, portfolio_returns.npy, weights.npy, dates.npy

The script stitches validation + test into a single continuous plot so
the reader can see the transition across splits.

Usage
-----
    python analysis/plot_training_curves.py --run_dir results/my_run
    python analysis/plot_training_curves.py --run_dir results/my_run \
        --output results/my_run/figures/tearsheet.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _drawdown_from_equity(equity: np.ndarray) -> np.ndarray:
    equity = np.asarray(equity, dtype=np.float64)
    if equity.size == 0:
        return equity
    running_max = np.maximum.accumulate(equity)
    return (running_max - equity) / np.where(running_max > 0, running_max, 1.0)


def _dates_to_datetime(dates_ns: np.ndarray):
    """Convert int64 ns timestamps to a numpy datetime array for plotting."""
    import pandas as pd
    return pd.to_datetime(np.asarray(dates_ns, dtype=np.int64), unit="ns").values


def build_tearsheet(
    output_path: Path,
    train: Mapping[str, np.ndarray] | None = None,
    val: Mapping[str, np.ndarray] | None = None,
    test: Mapping[str, np.ndarray] | None = None,
    title: str = "IQL Portfolio — Train / Val / Test Tearsheet",
) -> Path:
    """Render a tearsheet and save to ``output_path``.

    Each of ``train``/``val``/``test`` (when given) must be a mapping with
    keys ``equity``, ``returns``, ``dates``. Any of the three may be
    ``None`` (useful when the train split didn't produce an equity curve
    — the model is trained offline rather than rolled out on train data).

    Returns the path to the written file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    ax_eq, ax_dd, ax_cr = axes

    splits = {"train": train, "val": val, "test": test}
    colors = {"train": "#6c757d", "val": "#1f77b4", "test": "#d62728"}

    plotted_any = False
    for name, data in splits.items():
        if data is None:
            continue
        equity = np.asarray(data["equity"], dtype=np.float64)
        returns = np.asarray(data["returns"], dtype=np.float64)
        dates = _dates_to_datetime(data["dates"])
        if len(equity) == 0:
            continue
        plotted_any = True

        # Equity curve
        ax_eq.plot(dates, equity, color=colors[name], label=name, linewidth=1.4)
        ax_eq.axvspan(dates[0], dates[-1], color=colors[name], alpha=0.08)

        # Drawdown (plotted as a negative fill).
        dd = _drawdown_from_equity(equity)
        ax_dd.fill_between(dates, -dd, 0.0, color=colors[name], alpha=0.35,
                           label=name, linewidth=0.5)

        # Cumulative return
        cr = np.cumprod(1 + returns) - 1
        ax_cr.plot(dates, cr, color=colors[name], label=name, linewidth=1.4)

    if not plotted_any:
        raise ValueError(
            "build_tearsheet was called with no non-empty splits — "
            "there is nothing to plot."
        )

    ax_eq.set_title(title)
    ax_eq.set_ylabel("Equity (normalized)")
    ax_eq.grid(True, alpha=0.3)
    ax_eq.legend(loc="upper left")

    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.legend(loc="lower left")

    ax_cr.set_ylabel("Cumulative Return")
    ax_cr.set_xlabel("Date")
    ax_cr.grid(True, alpha=0.3)
    ax_cr.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_split(split_dir: Path) -> dict[str, np.ndarray] | None:
    """Load an artifact triple from ``split_dir``. Returns None if any
    mandatory file is missing."""
    eq = split_dir / "equity_curve.npy"
    rets = split_dir / "portfolio_returns.npy"
    dates = split_dir / "dates.npy"
    if not (eq.exists() and rets.exists() and dates.exists()):
        return None
    return {
        "equity": np.load(eq),
        "returns": np.load(rets),
        "dates": np.load(dates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train/val/test tearsheet.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory produced by scripts/train.py or "
                             "scripts/run_ablation.py.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path. Defaults to "
                             "<run_dir>/figures/tearsheet.png.")
    parser.add_argument("--title", type=str,
                        default="IQL Portfolio — Train / Val / Test Tearsheet")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    val = _load_split(run_dir / "validation")
    test = _load_split(run_dir / "test")

    if val is None and test is None:
        raise FileNotFoundError(
            f"No validation/ or test/ artifact triples found under {run_dir}. "
            f"Expected equity_curve.npy + portfolio_returns.npy + dates.npy."
        )

    output = Path(args.output) if args.output else run_dir / "figures" / "tearsheet.png"
    result = build_tearsheet(
        output_path=output,
        train=None,
        val=val,
        test=test,
        title=args.title,
    )
    print(f"Tearsheet saved → {result}")


if __name__ == "__main__":
    main()
