"""Aggregate metrics from all experiment runs into a single summary CSV.

Usage
-----
    python analysis/aggregate_results.py --results_dir results
    python analysis/aggregate_results.py --results_dir results/ablation

Scans all subdirectories for metrics.csv and config.yaml files, combines
them into a single DataFrame, and writes results/summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml


def collect_experiment_results(results_dir: Path) -> list[dict]:
    """Walk results_dir and collect metrics + config from each experiment.

    Each subdirectory is expected to contain:
    - metrics.csv: strategy-level metrics (IQL row is extracted)
    - config.yaml (optional): hyperparameters for the run

    Returns
    -------
    List of dicts, one per experiment.
    """
    rows = []

    for metrics_path in sorted(results_dir.rglob("metrics.csv")):
        exp_dir = metrics_path.parent
        experiment_id = exp_dir.name

        # Load config if available
        config = {}
        config_path = exp_dir / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

        # Read metrics.csv — extract IQL row
        with open(metrics_path, "r") as f:
            reader = csv.DictReader(f)
            for row_dict in reader:
                strategy = row_dict.get("strategy", "")
                if strategy == "IQL":
                    result = {
                        "experiment_id": config.get("experiment_id", experiment_id),
                        "expectile": config.get("expectile", ""),
                        "beta": config.get("beta", ""),
                        "transaction_cost": config.get("transaction_cost", ""),
                    }
                    # Add all metric columns
                    for key, val in row_dict.items():
                        if key != "strategy":
                            try:
                                result[key] = float(val)
                            except (ValueError, TypeError):
                                result[key] = val
                    rows.append(result)
                    break  # Only need the IQL row

    return rows


def write_summary(rows: list[dict], output_path: Path) -> None:
    """Write aggregated results to a CSV file."""
    if not rows:
        print("No experiment results found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine column order
    desired_order = [
        "experiment_id", "expectile", "beta", "transaction_cost",
        "sharpe_ratio", "max_drawdown", "cumulative_return",
        "annual_return", "annual_volatility", "turnover",
    ]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = [k for k in desired_order if k in all_keys]
    fieldnames += sorted(k for k in all_keys if k not in fieldnames)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written to {output_path} ({len(rows)} experiments)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Root directory to scan for metrics.csv files.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for summary.csv (default: <results_dir>/summary.csv).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "summary.csv"

    rows = collect_experiment_results(results_dir)
    write_summary(rows, output_path)


if __name__ == "__main__":
    main()
