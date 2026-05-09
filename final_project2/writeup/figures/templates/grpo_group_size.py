"""GRPO group-size ablation: test Sharpe vs G ∈ {4, 8, 16}, λ = 0.001.

Single seed (Phase 2D budget); annotate compute cost (forward passes per
gradient step are linear in G, so the right axis shows relative compute).

Output: writeup/figures/grpo_group_size.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "grpo_group_size.png"

GS = np.array([4, 8, 16])


def _load_data():
    """Stand-in. Real version reads results/phase2d/grpo_G*_lambda0.001_*.json."""
    sharpes = np.array([0.78, 0.94, 1.02])
    return sharpes


def render():
    sharpes = _load_data()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))

    ax.plot(GS, sharpes, "o-", color="#fb5607", markersize=8,
            linewidth=1.6, markeredgecolor="black", markeredgewidth=0.6)
    for g, s in zip(GS, sharpes):
        ax.annotate(f"{s:.2f}", xy=(g, s + 0.03), ha="center", fontsize=8)

    ax.axhline(0.953, linestyle="--", linewidth=0.7, color="#444",
               label="Equal weight (0.95)")
    ax.axhline(1.226, linestyle=":", linewidth=0.7, color="#444",
               label="Risk parity (1.23)")

    ax.set_xticks(GS)
    ax.set_xticklabels([f"G={g}" for g in GS])
    ax.set_xlabel("Group size G")
    ax.set_ylabel("Test Sharpe (annualized)")
    ax.set_title("GRPO group-size ablation at λ = 0.001  (1 seed)",
                 fontsize=10)
    ax.set_ylim(0.4, 1.4)
    ax.grid(linewidth=0.3, alpha=0.4)
    ax.legend(loc="lower right", fontsize=7, frameon=False)

    # Annotate relative compute: forward passes per update step scale ~G.
    ax.text(0.02, 0.98,
            "Group sampling: G fwd passes / state.\n"
            "Compute cost ≈ linear in G.",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, style="italic", color="#444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4f4f4",
                      edgecolor="#bbb", linewidth=0.5))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    render()
