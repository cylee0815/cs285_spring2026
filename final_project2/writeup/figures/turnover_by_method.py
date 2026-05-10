"""Turnover decomposition: average daily turnover by method, log scale.

Reads the aggregated CSVs (results/phase2_headline.csv,
results/phase2_appendix_leaky.csv, results/phase2a_causal/per_run.csv,
results/aux_iql_216d/iql_216d_seed42/metrics.json) and groups methods
into six families:

  1. Classical:        equal_weight, momentum_60d, risk_parity_60d
  2. Behavior-anchored offline (causal): bc, awac, cql_vanilla, iql
  3. Online-only:      online_only_sac_dirichlet, online_only_ppo_lstm,
                       online_only_grpo
  4. Naive O2O:        naive_o2o
  5. Adaptive O2O:     adaptive_o2o
  6. GRPO ablation:    grpo_G4, grpo_G8, grpo_G16
  7. Leaky-pipeline (appendix only): td3_bc, bcq

X-axis: ordered method labels.
Y-axis: log(turnover). Exact-zero bars rendered as hatched bars at a
small floor value with a "static" annotation, since log scale cannot
plot 0.

Output: writeup/figures/turnover_by_method.png (idempotent — re-runs
overwrite).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent / "turnover_by_method.png"

FLOOR = 1e-6        # log floor for plotting (must be > 0 for log scale)
STATIC_THRESH = 1e-5  # below this, the bar is rendered as "static" (hatched
                      # + annotated). Captures both literal-zero EW/SAC and
                      # functionally-zero naive_o2o (~2e-6).

# Family colors
COLORS = {
    "classical":         "#7f7f7f",
    "anchored_causal":   "#2ca02c",
    "online_only":       "#ff7f0e",
    "naive_o2o":         "#d62728",
    "adaptive_o2o":      "#9467bd",
    "grpo_ablation":     "#1f77b4",
    "leaky_appendix":    "#8c564b",
}

FAMILY_LABELS = {
    "classical":       "Classical",
    "anchored_causal": "Behavior-anchored (causal)",
    "online_only":     "Online-only",
    "naive_o2o":       "Naive O2O",
    "adaptive_o2o":    "Adaptive O2O",
    "grpo_ablation":   "GRPO ablation",
    "leaky_appendix":  "Leaky pipeline (appendix)",
}


def _load_headline() -> pd.DataFrame:
    p = ROOT / "results" / "phase2_headline.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _load_appendix_leaky() -> pd.DataFrame:
    p = ROOT / "results" / "phase2_appendix_leaky.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def main() -> int:
    headline = _load_headline()
    appendix = _load_appendix_leaky()

    bars = []  # list of dicts: label, family, turnover

    # Classical
    for s in ("equal_weight", "momentum_60d", "risk_parity_60d"):
        rows = headline[(headline["source"] == "classical_causal") &
                        (headline["method"] == s)]
        if not rows.empty:
            bars.append({"label": s.replace("_60d", ""),
                         "family": "classical",
                         "turnover": float(rows.iloc[0]["turnover_mean"])})

    # Behavior-anchored offline (causal)
    for a in ("bc", "awac", "cql_vanilla", "iql"):
        rows = headline[(headline["source"] == "phase2a_causal") &
                        (headline["method"] == a)]
        if not rows.empty:
            bars.append({"label": a.replace("cql_vanilla", "cql"),
                         "family": "anchored_causal",
                         "turnover": float(rows.iloc[0]["turnover_mean"])})

    # Online-only
    for m in ("online_only_sac_dirichlet", "online_only_ppo_lstm",
              "online_only_grpo"):
        rows = headline[headline["method"] == m]
        if not rows.empty:
            label = m.replace("online_only_", "").replace(
                "sac_dirichlet", "sac").replace("ppo_lstm", "ppo+lstm")
            bars.append({"label": label, "family": "online_only",
                         "turnover": float(rows.iloc[0]["turnover_mean"])})

    # Naive O2O
    rows = headline[(headline["source"] == "phase2c") &
                    (headline["method"] == "naive_o2o")]
    if not rows.empty:
        bars.append({"label": "naive O2O", "family": "naive_o2o",
                     "turnover": float(rows.iloc[0]["turnover_mean"])})

    # Adaptive O2O — only if rows exist (Batch 3 may not have landed)
    rows = headline[(headline["source"] == "phase2c") &
                    (headline["method"] == "adaptive_o2o")]
    if not rows.empty:
        bars.append({"label": "adaptive O2O", "family": "adaptive_o2o",
                     "turnover": float(rows.iloc[0]["turnover_mean"])})

    # GRPO ablation (3 G values)
    for g in (4, 8, 16):
        rows = headline[headline["method"] == f"grpo_G{g}"]
        if not rows.empty:
            bars.append({"label": f"GRPO G={g}", "family": "grpo_ablation",
                         "turnover": float(rows.iloc[0]["turnover_mean"])})

    # Leaky pipeline (appendix annotation: TD3+BC and BCQ)
    for a in ("td3_bc", "bcq"):
        rows = appendix[appendix["method"] == a]
        if not rows.empty:
            bars.append({"label": f"{a} (leaky)",
                         "family": "leaky_appendix",
                         "turnover": float(rows.iloc[0]["turnover_mean"])})

    if not bars:
        print("[warn] no bars to plot — aggregator may not have run")
        return 1

    # Plot
    n = len(bars)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(n)
    plotted_h = []
    static_marks = []
    for i, b in enumerate(bars):
        h = max(b["turnover"], FLOOR)
        is_static = b["turnover"] < STATIC_THRESH
        ax.bar(i, h, color=COLORS[b["family"]],
               edgecolor="black", linewidth=0.6,
               hatch="///" if is_static else None,
               alpha=0.85)
        plotted_h.append(h)
        if is_static:
            static_marks.append(i)
            ax.text(i, FLOOR * 1.6, "static\n(=0)", fontsize=7,
                    ha="center", va="bottom", color="black")

    # Reference lines
    ax.axhline(0.001, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(n - 0.5, 0.0011, "$\\lambda$=0.001", fontsize=7,
            color="gray", ha="right", va="bottom")
    ax.axhline(0.05, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(n - 0.5, 0.055, "active-trading threshold", fontsize=7,
            color="gray", ha="right", va="bottom")

    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2, 1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([b["label"] for b in bars], rotation=45,
                       ha="right", fontsize=9)
    ax.set_ylabel("avg daily turnover (log scale)")
    ax.set_title("Turnover decomposition: methods cluster as static-collapse "
                 "or active-trading;\n"
                 "active traders all underperform classical baselines "
                 "under $\\lambda{=}0.001$ friction.")
    ax.grid(True, axis="y", which="both", alpha=0.25)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS[k], label=FAMILY_LABELS[k])
        for k in ("classical", "anchored_causal", "online_only",
                  "naive_o2o", "adaptive_o2o", "grpo_ablation",
                  "leaky_appendix")
        if any(b["family"] == k for b in bars)
    ]
    if static_marks:
        legend_handles.append(mpatches.Patch(facecolor="white",
                                             edgecolor="black", hatch="///",
                                             label="static (turn = 0)"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9, ncol=2)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved -> {OUT}")
    print(f"  bars rendered: {n}")
    print(f"  static (=0): {[bars[i]['label'] for i in static_marks]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
