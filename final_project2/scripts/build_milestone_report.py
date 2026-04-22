"""Build CS285 milestone artifacts from IQL ablation runs.

Parses every run under ``runs/ablation/`` and produces:
    1. figures/iql_ablation_sharpe_curve.png
    2. figures/iql_ablation_summary.png
    3. figures/ablation_summary.csv

The script is intentionally dependency-light: stdlib + numpy + pandas + matplotlib.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = PROJECT_ROOT / "runs" / "ablation"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RunRecord:
    experiment_name: str
    run_dir: Path
    tau: float
    beta: float
    transaction_cost: float
    val_curve: Optional[pd.DataFrame]
    best_val_sharpe: Optional[float]
    final_test_sharpe: Optional[float]
    cumulative_return: Optional[float]
    max_drawdown: Optional[float]


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return yaml.safe_load(f) or {}


def _load_val_curve(run_dir: Path) -> Optional[pd.DataFrame]:
    path = run_dir / "validation.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "step" not in df.columns or "sharpe_ratio" not in df.columns:
        return None
    return df


def _load_test_metrics(run_dir: Path) -> dict:
    path = run_dir / "test" / "metrics.csv"
    if not path.exists():
        return {}
    out: dict = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[row["metric"]] = float(row["value"])
            except (KeyError, ValueError):
                continue
    return out


def _latest_timestamp_dir(exp_dir: Path) -> Optional[Path]:
    subdirs = [p for p in exp_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: p.name)
    return subdirs[-1]


def collect_runs() -> list[RunRecord]:
    records: list[RunRecord] = []
    for exp_dir in sorted(ABLATION_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        run_dir = _latest_timestamp_dir(exp_dir)
        if run_dir is None:
            continue
        cfg = _load_config(run_dir)

        val_curve = _load_val_curve(run_dir)
        best_val_sharpe = (
            float(val_curve["sharpe_ratio"].max()) if val_curve is not None else None
        )
        test_metrics = _load_test_metrics(run_dir)

        records.append(
            RunRecord(
                experiment_name=exp_dir.name,
                run_dir=run_dir,
                tau=float(cfg.get("expectile", float("nan"))),
                beta=float(cfg.get("beta", float("nan"))),
                transaction_cost=float(cfg.get("transaction_cost", float("nan"))),
                val_curve=val_curve,
                best_val_sharpe=best_val_sharpe,
                final_test_sharpe=test_metrics.get("sharpe_ratio"),
                cumulative_return=test_metrics.get("cumulative_return"),
                max_drawdown=test_metrics.get("max_drawdown"),
            )
        )
    return records


def build_summary(records: list[RunRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "experiment_name": r.experiment_name,
                "tau": r.tau,
                "beta": r.beta,
                "transaction_cost": r.transaction_cost,
                "best_validation_sharpe": r.best_val_sharpe
                if r.best_val_sharpe is not None
                else "N/A",
                "final_test_sharpe": r.final_test_sharpe
                if r.final_test_sharpe is not None
                else "N/A",
                "cumulative_return": r.cumulative_return
                if r.cumulative_return is not None
                else "N/A",
                "max_drawdown": r.max_drawdown
                if r.max_drawdown is not None
                else "N/A",
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["transaction_cost", "tau", "beta"]).reset_index(drop=True)
    return df


def plot_sharpe_curves(records: list[RunRecord], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    plotted = 0
    for r in records:
        if r.val_curve is None:
            continue
        label = f"tau={r.tau:g}, beta={r.beta:g}, tc={r.transaction_cost:g}"
        ax.plot(
            r.val_curve["step"].to_numpy(),
            r.val_curve["sharpe_ratio"].to_numpy(),
            label=label,
            linewidth=1.2,
            alpha=0.85,
        )
        plotted += 1
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation Sharpe ratio")
    ax.set_title(f"IQL ablation: validation Sharpe over training ({plotted} runs)")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(df: pd.DataFrame, out_path: Path) -> None:
    numeric = df.copy()
    numeric["final_test_sharpe"] = pd.to_numeric(
        numeric["final_test_sharpe"], errors="coerce"
    )
    numeric["best_validation_sharpe"] = pd.to_numeric(
        numeric["best_validation_sharpe"], errors="coerce"
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(numeric))
    width = 0.4

    ax.bar(x - width / 2, numeric["best_validation_sharpe"].to_numpy(),
           width=width, label="Best validation Sharpe")
    ax.bar(x + width / 2, numeric["final_test_sharpe"].to_numpy(),
           width=width, label="Test Sharpe (final policy)")

    labels = [
        f"tau={row.tau:g}\nbeta={row.beta:g}\ntc={row.transaction_cost:g}"
        for row in numeric.itertuples(index=False)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("IQL ablation: validation vs. test Sharpe per configuration")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    records = collect_runs()
    if not records:
        raise SystemExit(f"No runs found under {ABLATION_DIR}")

    df = build_summary(records)

    summary_csv = FIGURES_DIR / "ablation_summary.csv"
    df.to_csv(summary_csv, index=False)

    sharpe_curve_path = FIGURES_DIR / "iql_ablation_sharpe_curve.png"
    summary_png_path = FIGURES_DIR / "iql_ablation_summary.png"
    plot_sharpe_curves(records, sharpe_curve_path)
    plot_summary_bar(df, summary_png_path)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", lambda v: f"{v:.4f}",
    ):
        print(df.to_string(index=False))

    print()
    print(f"Saved: {summary_csv}")
    print(f"Saved: {sharpe_curve_path}")
    print(f"Saved: {summary_png_path}")


if __name__ == "__main__":
    main()
