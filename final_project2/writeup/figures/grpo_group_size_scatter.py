"""Phase 2D 9-point scatter (3 seeds x 3 G) with seed-color coding and
mean lines.

Reads results/phase2d/grpo_G{4,8,16}_lambda0.001_seed{42,1337,2024}/metrics.json
and writes writeup/figures/grpo_group_size_scatter.png.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase2d"
OUT = Path(__file__).parent / "grpo_group_size_scatter.png"

GS = [4, 8, 16]
SEEDS = [42, 1337, 2024]
COLORS = {42: "#1f77b4", 1337: "#d62728", 2024: "#2ca02c"}

data = {(g, s): None for g in GS for s in SEEDS}
for g in GS:
    for s in SEEDS:
        f = RESULTS / f"grpo_G{g}_lambda0.001_seed{s}" / "metrics.json"
        with f.open() as fh:
            data[(g, s)] = json.load(fh)["test"]["sharpe_ratio"]

fig, ax = plt.subplots(figsize=(6.5, 4.0))
for s in SEEDS:
    ys = [data[(g, s)] for g in GS]
    ax.plot(GS, ys, "o-", color=COLORS[s], label=f"seed {s}",
            markersize=8, linewidth=1.5, alpha=0.85)

means = [np.mean([data[(g, s)] for s in SEEDS]) for g in GS]
stds = [np.std([data[(g, s)] for s in SEEDS], ddof=1) for g in GS]
ax.errorbar(GS, means, yerr=stds, color="black", linewidth=2.0,
            fmt="D-", markersize=9, capsize=5,
            label=r"mean $\pm$ 1 std (n=3)", zorder=10)

# Equal-weight reference line
ax.axhline(0.953, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(15.5, 0.973, "EW = 0.953", fontsize=8, color="gray", ha="right")

ax.set_xscale("log", base=2)
ax.set_xticks(GS)
ax.set_xticklabels([str(g) for g in GS])
ax.set_xlabel("group size $G$ (log scale)")
ax.set_ylabel("test Sharpe ratio")
ax.set_title("Phase 2D: GRPO test Sharpe vs.\\ group size, 3 seeds")
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"saved -> {OUT}")
print()
print("Summary (mean ± std, n=3):")
for g, m, sd in zip(GS, means, stds):
    print(f"  G={g:2d}  Sharpe = {m:+.3f} ± {sd:.3f}  range [{min(data[(g,s)] for s in SEEDS):+.3f}, {max(data[(g,s)] for s in SEEDS):+.3f}]")
