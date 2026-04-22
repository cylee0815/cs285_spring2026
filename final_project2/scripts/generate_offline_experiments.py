#!/usr/bin/env python3
"""
Generate the full offline RL experiment matrix.

Outputs one command per line to experiments/offline_experiments.txt.
Does NOT execute anything.

Usage:
    python scripts/generate_offline_experiments.py
"""
import os
import itertools

# 10 active algorithms (MBPO/MOPO disabled)
ALGORITHMS = [
    "bc", "fisher_bc", "td3_bc", "awac", "cql_vanilla",
    "iql", "edac", "bcq",
    "decision_transformer", "trajectory_transformer",
]

N_STEPS = [1, 3, 5]
SEEDS = [0, 1, 2]

SCRIPT = "python offline_rl/scripts/run_offline.py"


def main():
    lines = []
    for algo, n_step, seed in itertools.product(ALGORITHMS, N_STEPS, SEEDS):
        cmd = (
            f"{SCRIPT} --base_config={algo} --n_step={n_step} "
            f"--seed={seed} --run_group=offline_matrix"
        )
        lines.append(cmd)

    out_dir = os.path.join("experiments")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "offline_experiments.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated {len(lines)} experiment commands -> {out_path}")
    print(f"  Algorithms : {len(ALGORITHMS)}")
    print(f"  n_step     : {N_STEPS}")
    print(f"  Seeds      : {SEEDS}")
    print(f"  Total runs : {len(lines)}")


if __name__ == "__main__":
    main()
