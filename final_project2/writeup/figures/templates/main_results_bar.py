"""Headline four-way comparison bar chart.

Renders per-condition test Sharpe with seed-error bars at lambda=0.001.
Dummy data; replace ``_load_data`` with a read of
``results/phase2_summary.csv`` once Phase 2C completes.

Output: writeup/figures/main_results_bar.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "main_results_bar.png"

CONDITIONS = [
    "Frozen offline\n(best Phase 2A)",
    "Online-only SAC\n(Phase 2B)",
    "Naive fine-tune\n(no β schedule)",
    "Adaptive O2O\n(β = α·σ(η·KL))",
]
COLORS = ["#888888", "#3a86ff", "#ffb703", "#fb5607"]


def _load_data():
    """Stand-in. Real version reads phase2_summary.csv."""
    rng = np.random.default_rng(0)
    means = np.array([0.32, 0.85, 0.65, 1.05])
    stds = rng.uniform(0.05, 0.18, size=4)
    return means, stds


def render():
    means, stds = _load_data()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    xs = np.arange(len(CONDITIONS))
    bars = ax.bar(xs, means, yerr=stds, capsize=4,
                  color=COLORS, edgecolor="black", linewidth=0.8, alpha=0.92)

    # Bench lines: equal-weight and risk-parity test-window Sharpes
    # (from runs/ablation/.../metrics.json).
    ax.axhline(0.953, linestyle="--", linewidth=0.8, color="#444",
               label="Equal weight (0.95)")
    ax.axhline(1.226, linestyle=":", linewidth=0.8, color="#444",
               label="Risk parity (1.23)")

    for bar, m, s in zip(bars, means, stds):
        ax.annotate(f"{m:.2f}±{s:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, m + s + 0.04),
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(CONDITIONS, fontsize=8)
    ax.set_ylabel("Test Sharpe (annualized)")
    ax.set_title("Four-way O2O comparison at λ = 0.001  (3 seeds, mean ± std)",
                 fontsize=10)
    ax.set_ylim(-0.2, 1.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    render()
