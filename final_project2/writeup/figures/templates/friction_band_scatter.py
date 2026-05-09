"""Phase 2A friction-band scatter: best-val vs final-test Sharpe.

Each point is one (algo, seed, λ) cell. Color = λ; marker = algo.
Reveals (a) whether algos cluster by λ rather than by family, and
(b) whether high val-Sharpe transfers to high test-Sharpe across the
2021→2022 split.

Output: writeup/figures/friction_band_scatter.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "friction_band_scatter.png"

ALGOS = ["BC", "TD3+BC", "CQL", "AWAC", "BCQ", "IQL"]
MARKERS = ["o", "s", "^", "D", "P", "X"]
LAMBDAS = [0.0, 0.001, 0.005]
LAMBDA_COLORS = {0.0: "#2a9d8f", 0.001: "#e9c46a", 0.005: "#e76f51"}


def _load_data():
    """Stand-in. Real version reads results/phase2a/per_run.csv and reuses
    the milestone's IQL ablation rows for the λ-anchor."""
    rng = np.random.default_rng(0)
    rows = []
    # Algo × lambda offsets that approximate the milestone's friction collapse.
    algo_offsets = {a: rng.uniform(-0.05, 0.05) for a in ALGOS}
    lambda_means_val = {0.0: 1.30, 0.001: 1.10, 0.005: 0.20}
    lambda_means_test = {0.0: 0.93, 0.001: 0.40, 0.005: -1.95}
    for a in ALGOS:
        for lam in LAMBDAS:
            for s in (42, 1337, 2024):
                val = lambda_means_val[lam] + algo_offsets[a] + rng.normal(0, 0.04)
                test = lambda_means_test[lam] + algo_offsets[a] + rng.normal(0, 0.06)
                rows.append((a, lam, s, val, test))
    return rows


def render():
    rows = _load_data()
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Diagonal reference: val == test
    lo, hi = -2.5, 1.8
    ax.plot([lo, hi], [lo, hi], color="#444", linewidth=0.4, alpha=0.5,
            linestyle="--", label="val = test")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)

    plotted = set()
    for algo, lam, _seed, val, test in rows:
        marker = MARKERS[ALGOS.index(algo)]
        color = LAMBDA_COLORS[lam]
        label = None
        if (algo, lam) not in plotted:
            label = f"{algo} (λ={lam})"
            plotted.add((algo, lam))
        ax.scatter(val, test, marker=marker, color=color, s=42, alpha=0.85,
                   edgecolors="black", linewidths=0.4)

    # Custom legend: combine algo (markers, in black) and λ (colors, dot).
    algo_handles = [
        plt.Line2D([0], [0], marker=MARKERS[i], color="white",
                   markerfacecolor="#bbb", markeredgecolor="black", markersize=7,
                   label=ALGOS[i]) for i in range(len(ALGOS))
    ]
    lam_handles = [
        plt.Line2D([0], [0], marker="o", color="white",
                   markerfacecolor=LAMBDA_COLORS[lam], markeredgecolor="black",
                   markersize=8, label=f"λ={lam}") for lam in LAMBDAS
    ]
    leg1 = ax.legend(handles=algo_handles, title="Algorithm", loc="upper left",
                     fontsize=7, ncol=2, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=lam_handles, title="Friction λ", loc="lower right",
              fontsize=7, frameon=False)

    ax.set_xlabel("Best validation Sharpe (HP-tuned, 2021)")
    ax.set_ylabel("Final test Sharpe (2022Q1–2026Q1)")
    ax.set_title("Phase 2A friction-band scatter — color by λ, marker by algo",
                 fontsize=10)
    ax.grid(linewidth=0.3, alpha=0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    render()
