"""Aggregate per-run metrics.json files into per-phase CSVs + a summary CSV
+ a headline cell-table.

Reads:
    results/phase2a/<run_name>/metrics.json          (offline runs, leaky pipeline)
    results/phase2a_causal/<run_name>/metrics.json   (offline runs, causal pipeline)
    results/phase2b/<run_name>/metrics.json          (online + GRPO runs, causal)
    results/phase2c/<run_name>/metrics.json          (O2O runs, causal)
    results/phase2d/<run_name>/metrics.json          (GRPO group-size ablation, causal)
    results/classical_causal.csv                     (classical baselines)

Writes:
    results/phase2{a,a_causal,b,c,d}/per_run.csv
    results/phase2_summary.csv     (concatenation, with phase column)
    results/phase2_headline.csv    (per-(method,lambda) cells for the main
                                    results table — pulls behavior-anchored
                                    rows from phase2a_causal, online/GRPO
                                    from phase2b, O2O from phase2c, GRPO
                                    ablation from phase2d, classical from
                                    results/classical_causal.csv)
    results/phase2_appendix_leaky.csv  (per-(method,lambda) leaky-pipeline
                                    cells for the appendix table — TD3+BC,
                                    BCQ, and the four behavior-anchored
                                    algos pre-fix)

Run-name convention: ``{algo}_lambda{tc}_seed{seed}`` so the columns can be
parsed back without depending on metrics.json fields.
Phase 2D run-name convention: ``grpo_G{group}_lambda{tc}_seed{seed}``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

NAME_RE = re.compile(r"^(?P<algo>[a-z0-9_]+?)_lambda(?P<tc>[0-9.]+)_seed(?P<seed>\d+)$")
GRPO_ABL_NAME_RE = re.compile(
    r"^grpo_G(?P<group_size>\d+)_lambda(?P<tc>[0-9.]+)_seed(?P<seed>\d+)$"
)


def parse_name(run_name: str) -> dict:
    """Extract algo / lambda / seed (and optional group_size) from a
    Phase 2 run-name. Falls back to just `algo=run_name` if the convention
    isn't followed."""
    m = GRPO_ABL_NAME_RE.match(run_name)
    if m:
        return {
            "algo": "grpo",
            "group_size": int(m.group("group_size")),
            "lambda": float(m.group("tc")),
            "seed": int(m.group("seed")),
        }
    m = NAME_RE.match(run_name)
    if m:
        return {
            "algo": m.group("algo"),
            "group_size": None,
            "lambda": float(m.group("tc")),
            "seed": int(m.group("seed")),
        }
    return {"algo": run_name, "group_size": None, "lambda": None, "seed": None}


def collect_phase(phase_dir: Path) -> pd.DataFrame:
    rows = []
    if not phase_dir.is_dir():
        return pd.DataFrame()
    for run_dir in sorted(phase_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        mfile = run_dir / "metrics.json"
        if not mfile.exists():
            continue
        try:
            with mfile.open() as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"[warn] could not parse {mfile}", file=sys.stderr)
            continue
        info = parse_name(run_dir.name)
        # Pull algo/seed from metrics.json when present (more authoritative
        # than the directory name).
        info["algo"] = data.get("algo", info["algo"])
        info["seed"] = data.get("seed", info["seed"])
        info["lambda"] = data.get("transaction_cost", info["lambda"])
        info["run_name"] = run_dir.name
        info["n_step"] = data.get("n_step")
        info["behavior_mix"] = data.get("behavior_mix")
        info["best_val_sharpe"] = data.get("best_val_sharpe")
        info["best_val_step"] = data.get("best_val_step")
        test = data.get("test", {})
        info["test_sharpe"] = test.get("sharpe_ratio")
        info["test_annual_return"] = test.get("annual_return")
        info["test_annual_volatility"] = test.get("annual_volatility")
        info["test_max_drawdown"] = test.get("max_drawdown")
        info["test_turnover"] = test.get("turnover")
        info["test_cumulative_return"] = test.get("cumulative_return")
        info["test_sortino"] = test.get("sortino_ratio")
        info["test_calmar"] = test.get("calmar_ratio")
        rows.append(info)
    return pd.DataFrame(rows)


_METRIC_COLS = [
    "test_sharpe", "test_cumulative_return", "test_max_drawdown",
    "test_turnover", "test_annual_return", "test_annual_volatility",
]


def _aggregate_cells(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Reduce per-run rows to per-(method, lambda[, group_size]) cells with
    mean/std/n across seeds. Robust to NaN seeds (legacy runs)."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)
    out = g[_METRIC_COLS].agg(["mean", "std", "count"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    return out.reset_index()


def _build_headline(
    causal_a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame, d: pd.DataFrame,
    classical_csv: Path,
) -> pd.DataFrame:
    """Build the main results table, one row per (source, method).

    Behavior-anchored offline algos pull from causal Phase 2A.
    Online-only algos pull from Phase 2B.
    O2O conditions pull from Phase 2C (when available).
    GRPO ablation pulls from Phase 2D (one row per G).
    Classical baselines pulled from a precomputed CSV (no seeds).
    TD3+BC and BCQ are intentionally omitted from headline (appendix only).
    """
    rows = []
    # Causal Phase 2A: behavior-anchored algos at lambda=0.001.
    if not causal_a.empty:
        keep = causal_a[
            causal_a["algo"].isin(["bc", "awac", "cql_vanilla", "iql"])
            & (causal_a["lambda"] == 0.001)
        ]
        cells = _aggregate_cells(keep, ["algo", "lambda"])
        for _, r in cells.iterrows():
            rows.append({
                "source": "phase2a_causal",
                "method": r["algo"],
                "lambda": r["lambda"],
                "group_size": None,
                "n_seeds": int(r["test_sharpe_count"]),
                "sharpe_mean": r["test_sharpe_mean"],
                "sharpe_std": r["test_sharpe_std"],
                "cum_return_mean": r["test_cumulative_return_mean"],
                "max_dd_mean": r["test_max_drawdown_mean"],
                "turnover_mean": r["test_turnover_mean"],
            })
    # Phase 2B: online-only at lambda=0.001.
    if not b.empty:
        keep = b[b["lambda"] == 0.001]
        cells = _aggregate_cells(keep, ["algo", "lambda"])
        for _, r in cells.iterrows():
            rows.append({
                "source": "phase2b",
                "method": f"online_only_{r['algo']}",
                "lambda": r["lambda"],
                "group_size": None,
                "n_seeds": int(r["test_sharpe_count"]),
                "sharpe_mean": r["test_sharpe_mean"],
                "sharpe_std": r["test_sharpe_std"],
                "cum_return_mean": r["test_cumulative_return_mean"],
                "max_dd_mean": r["test_max_drawdown_mean"],
                "turnover_mean": r["test_turnover_mean"],
            })
    # Phase 2C: O2O conditions at lambda=0.001 (when available).
    if not c.empty:
        keep = c[c["lambda"] == 0.001]
        cells = _aggregate_cells(keep, ["algo", "lambda"])
        for _, r in cells.iterrows():
            rows.append({
                "source": "phase2c",
                "method": r["algo"],
                "lambda": r["lambda"],
                "group_size": None,
                "n_seeds": int(r["test_sharpe_count"]),
                "sharpe_mean": r["test_sharpe_mean"],
                "sharpe_std": r["test_sharpe_std"],
                "cum_return_mean": r["test_cumulative_return_mean"],
                "max_dd_mean": r["test_max_drawdown_mean"],
                "turnover_mean": r["test_turnover_mean"],
            })
    # Phase 2D: GRPO ablation (one row per G).
    if not d.empty:
        cells = _aggregate_cells(d, ["algo", "group_size", "lambda"])
        for _, r in cells.iterrows():
            rows.append({
                "source": "phase2d",
                "method": f"grpo_G{int(r['group_size'])}",
                "lambda": r["lambda"],
                "group_size": int(r["group_size"]),
                "n_seeds": int(r["test_sharpe_count"]),
                "sharpe_mean": r["test_sharpe_mean"],
                "sharpe_std": r["test_sharpe_std"],
                "cum_return_mean": r["test_cumulative_return_mean"],
                "max_dd_mean": r["test_max_drawdown_mean"],
                "turnover_mean": r["test_turnover_mean"],
            })
    # Classical baselines (no seeds, single sweep).
    if classical_csv.exists():
        cls = pd.read_csv(classical_csv)
        for _, r in cls.iterrows():
            rows.append({
                "source": "classical_causal",
                "method": r["strategy"],
                "lambda": 0.001,
                "group_size": None,
                "n_seeds": 1,
                "sharpe_mean": r["sharpe_ratio"],
                "sharpe_std": float("nan"),
                "cum_return_mean": r["cumulative_return"],
                "max_dd_mean": r["max_drawdown"],
                "turnover_mean": r["avg_daily_turnover"],
            })
    return pd.DataFrame(rows)


def _build_appendix_leaky(leaky_a: pd.DataFrame) -> pd.DataFrame:
    """Build the appendix table: leaky Phase 2A cells for all six algos.

    TD3+BC and BCQ are the headline rows of this table (Sharpe 6.8,
    3.9 — leak demonstration). The four behavior-anchored algos provide
    the contrast. Pre-fix data, do not present as headline numbers.
    """
    if leaky_a.empty:
        return pd.DataFrame()
    cells = _aggregate_cells(leaky_a, ["algo", "lambda"])
    out_rows = []
    for _, r in cells.iterrows():
        out_rows.append({
            "source": "phase2a_LEAKY",
            "method": r["algo"],
            "lambda": r["lambda"],
            "n_seeds": int(r["test_sharpe_count"]),
            "sharpe_mean": r["test_sharpe_mean"],
            "sharpe_std": r["test_sharpe_std"],
            "cum_return_mean": r["test_cumulative_return_mean"],
            "max_dd_mean": r["test_max_drawdown_mean"],
            "turnover_mean": r["test_turnover_mean"],
        })
    return pd.DataFrame(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    args = parser.parse_args()
    root = Path(args.results_root)

    summary_frames = []
    phase_dfs: dict[str, pd.DataFrame] = {}
    for phase in ("phase2a", "phase2a_causal", "phase2b", "phase2c", "phase2d"):
        df = collect_phase(root / phase)
        out = root / phase / "per_run.csv"
        if df.empty:
            print(f"[{phase}] no runs found at {root / phase}")
            phase_dfs[phase] = df
            continue
        sort_cols = [c for c in ["algo", "lambda", "group_size", "seed"]
                     if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        df.to_csv(out, index=False)
        df_sum = df.assign(phase=phase)
        summary_frames.append(df_sum)
        phase_dfs[phase] = df
        print(f"[{phase}] {len(df)} runs -> {out}")

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        out = root / "phase2_summary.csv"
        summary.to_csv(out, index=False)
        print(f"[summary] {len(summary)} runs -> {out}")

    headline = _build_headline(
        causal_a=phase_dfs.get("phase2a_causal", pd.DataFrame()),
        b=phase_dfs.get("phase2b", pd.DataFrame()),
        c=phase_dfs.get("phase2c", pd.DataFrame()),
        d=phase_dfs.get("phase2d", pd.DataFrame()),
        classical_csv=root / "classical_causal.csv",
    )
    if not headline.empty:
        out = root / "phase2_headline.csv"
        headline.to_csv(out, index=False)
        print(f"[headline] {len(headline)} cells -> {out}")
    else:
        print("[headline] no rows assembled (check causal/online inputs)")

    appendix = _build_appendix_leaky(phase_dfs.get("phase2a", pd.DataFrame()))
    if not appendix.empty:
        out = root / "phase2_appendix_leaky.csv"
        appendix.to_csv(out, index=False)
        print(f"[appendix] {len(appendix)} cells -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
