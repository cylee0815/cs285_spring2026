"""Generate the figures the Phase 2 report needs from the per-run CSVs.

Reads:
    results/phase2_summary.csv  (produced by aggregate_phase2.py)
    results/phase2c/<run>/cql_weight_traj.npy  (for adaptive β_t)

Writes (best-effort — skips a figure if the underlying data is absent):
    figures/phase2/four_way_comparison.png
    figures/phase2/turnover_comparison.png
    figures/phase2/adaptive_beta_trajectory.png
    figures/phase2/friction_band_comparison.png
    figures/phase2/grpo_group_size_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("figures/phase2")


def _ensure_out():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _bar_with_err(ax, df, x_label_col, value_col, ylabel, title, color_col=None):
    """df has columns x_label_col, value_col, color_col (optional)."""
    grouped = df.groupby(x_label_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    grouped["sem"] = grouped["std"] / grouped["count"].clip(lower=1).pow(0.5)
    xs = np.arange(len(grouped))
    ax.bar(xs, grouped["mean"], yerr=grouped["sem"], capsize=3,
           color="tab:blue", alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(grouped[x_label_col], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.5)


def four_way_comparison(summary: pd.DataFrame):
    """Four conditions × test Sharpe, grouped by lambda. Reads from summary."""
    if summary.empty or "phase" not in summary.columns:
        return
    # Build the four conditions per the prompt.
    df_records = []
    # (1) Frozen offline: Phase 2A's IQL/CQL test rows.
    a = summary[summary["phase"] == "phase2a"]
    if not a.empty:
        # Pick the best offline algo per (seed, λ): max test_sharpe.
        best_offline = (a.sort_values("test_sharpe", ascending=False)
                          .groupby(["seed", "lambda"], dropna=False).head(1))
        for _, r in best_offline.iterrows():
            df_records.append({
                "condition": "frozen_offline",
                "seed": r["seed"], "lambda": r["lambda"],
                "test_sharpe": r["test_sharpe"], "test_turnover": r.get("test_turnover"),
            })
    # (2) Online-only SAC: Phase 2B sac_dirichlet rows.
    b = summary[(summary["phase"] == "phase2b") & (summary["algo"] == "sac_dirichlet")]
    for _, r in b.iterrows():
        df_records.append({
            "condition": "online_only_sac",
            "seed": r["seed"], "lambda": r["lambda"],
            "test_sharpe": r["test_sharpe"], "test_turnover": r.get("test_turnover"),
        })
    # (3) and (4): Phase 2C runs.
    c = summary[summary["phase"] == "phase2c"]
    for _, r in c.iterrows():
        cond = "naive_finetune" if r["algo"].startswith("naive") else "adaptive_o2o"
        df_records.append({
            "condition": cond,
            "seed": r["seed"], "lambda": r["lambda"],
            "test_sharpe": r["test_sharpe"], "test_turnover": r.get("test_turnover"),
        })
    if not df_records:
        return
    df = pd.DataFrame(df_records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _bar_with_err(axes[0], df, "condition", "test_sharpe",
                  "Test Sharpe (annualized)",
                  "Phase 2C — Four-way comparison (λ=0.001)")
    _bar_with_err(axes[1], df, "condition", "test_turnover",
                  "Avg daily turnover",
                  "Phase 2C — Turnover by condition")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "four_way_comparison.png", dpi=150)
    plt.close(fig)
    df.to_csv(OUT_DIR / "four_way_comparison_data.csv", index=False)


def adaptive_beta(summary: pd.DataFrame):
    """β_t (cql_weight) trajectory for adaptive O2O runs."""
    if summary.empty or "phase" not in summary.columns:
        return
    c = summary[(summary["phase"] == "phase2c") & summary["algo"].astype(str).str.startswith("adaptive")]
    if c.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for _, r in c.iterrows():
        traj_path = Path("results/phase2c") / r["run_name"] / "cql_weight_traj.npy"
        if not traj_path.exists():
            continue
        traj = np.load(traj_path)
        if traj.size == 0:
            continue
        ax.plot(traj, alpha=0.6, label=f"seed={r['seed']}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Online training step (post-warmup)")
    ax.set_ylabel("Adaptive cql_weight (β_t)")
    ax.set_title("Phase 2C — Adaptive β_t over online fine-tuning (λ=0.001)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "adaptive_beta_trajectory.png", dpi=150)
    plt.close(fig)


def friction_band(summary: pd.DataFrame):
    """Best-val vs test Sharpe across (algo, λ) — Phase 2A only."""
    if summary.empty or "phase" not in summary.columns:
        return
    a = summary[summary["phase"] == "phase2a"].dropna(subset=["test_sharpe"])
    if a.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    algos = sorted(a["algo"].unique())
    markers = ["o", "s", "^", "D", "P", "X", "*", "v"]
    cmap = {0.0: "tab:green", 0.001: "tab:orange", 0.005: "tab:red"}
    for i, algo in enumerate(algos):
        sub = a[a["algo"] == algo]
        for tc, color in cmap.items():
            row = sub[np.isclose(sub["lambda"].fillna(-1), tc)]
            if row.empty:
                continue
            ax.scatter(row["best_val_sharpe"], row["test_sharpe"],
                       marker=markers[i % len(markers)], color=color,
                       label=f"{algo} λ={tc}", s=60, alpha=0.8, edgecolors="black",
                       linewidths=0.5)
    ax.plot([-2, 3], [-2, 3], color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Best validation Sharpe")
    ax.set_ylabel("Final test Sharpe")
    ax.set_title("Phase 2A — Friction-band scatter (color=λ, marker=algo)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "friction_band_comparison.png", dpi=150)
    plt.close(fig)


def grpo_group_size(grpo_dir: Path):
    """GRPO group-size ablation: test Sharpe vs G."""
    rows = []
    for run_dir in sorted(grpo_dir.iterdir()) if grpo_dir.is_dir() else []:
        m = run_dir / "metrics.json"
        if not m.exists():
            continue
        import json
        with m.open() as f:
            d = json.load(f)
        # Run name encodes G via "_G{n}_".
        import re
        match = re.search(r"_G(\d+)_", run_dir.name)
        if not match:
            continue
        rows.append({
            "G": int(match.group(1)),
            "test_sharpe": d.get("test", {}).get("sharpe_ratio"),
        })
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("G")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["G"], df["test_sharpe"], "o-", color="tab:blue")
    ax.set_xlabel("Group size G")
    ax.set_ylabel("Test Sharpe (annualized)")
    ax.set_title("Phase 2D — GRPO group-size ablation (λ=0.001)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "grpo_group_size_curve.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/phase2_summary.csv")
    parser.add_argument("--phase2d_dir", default="results/phase2d")
    args = parser.parse_args()

    _ensure_out()
    summary = pd.DataFrame()
    if Path(args.summary).exists():
        summary = pd.read_csv(args.summary)
        print(f"loaded summary: {len(summary)} rows from {args.summary}")
    else:
        print(f"[warn] {args.summary} not found — most figures will be skipped")

    four_way_comparison(summary)
    adaptive_beta(summary)
    friction_band(summary)
    grpo_group_size(Path(args.phase2d_dir))

    print(f"figures written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
