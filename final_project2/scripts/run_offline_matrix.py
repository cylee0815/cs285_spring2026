"""Generate the offline-RL benchmark experiment grid.

# Environment: all runs use env/portfolio_env.py (the project's canonical env).

Prints the command lines for the 108-run study (12 algorithms × 3 n_step ×
3 seeds) without executing them. Redirect to a file and feed into your
cluster launcher (sbatch, modal, etc.).

Usage:
    uv run python scripts/run_offline_matrix.py                 # prints to stdout
    uv run python scripts/run_offline_matrix.py --run_group=offline_nstep > jobs.txt
    uv run python scripts/run_offline_matrix.py --dry_run=false # would execute (guard-rail: still prints, still does NOT execute)
    uv run python scripts/run_offline_matrix.py --algorithms bc iql --seeds 0
        # restrict the grid to a subset
"""

from __future__ import annotations

import argparse
import itertools
import shlex
import sys


ALGORITHMS = [
    "bc",
    "fisher_bc",
    "td3_bc",
    "awac",
    "cql_vanilla",  # maps to "CQL" in the research objective
    "iql",
    "edac",
    "bcq",
    "mbpo",
    "mopo",
    "decision_transformer",
    "trajectory_transformer",
]

N_STEPS = [1, 3, 5]
SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run_group", default="offline_baselines_nstep",
        help="WandB group name shared across all runs in the matrix.",
    )
    p.add_argument(
        "--algorithms", nargs="+", default=ALGORITHMS,
        choices=ALGORITHMS,
        help="Subset of algorithms to include (default: all 12).",
    )
    p.add_argument(
        "--n_steps", nargs="+", type=int, default=N_STEPS,
        help="Subset of n_step values (default: 1 3 5).",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Subset of seeds (default: 0 1 2).",
    )
    p.add_argument(
        "--transaction_cost", type=float, default=0.001,
        help="λ on ||w_t - w_{t-1}||_1. The research question is whether "
             "multi-step returns help under friction — set >0 for the real study.",
    )
    p.add_argument(
        "--offline_data_steps", type=int, default=50_000,
        help="Behavior-policy rollouts used to fill the offline buffer.",
    )
    p.add_argument(
        "--n_offline_updates", type=int, default=None,
        help="Override config.n_offline_updates (leave None to use each "
             "algorithm's default).",
    )
    p.add_argument(
        "--python", default="uv run python",
        help="Python invocation prefix (so the same script can emit commands "
             "for local `python` or cluster wrappers).",
    )
    p.add_argument(
        "--prefix", default="",
        help="String prepended to every emitted command (e.g. a sbatch wrapper).",
    )
    return p.parse_args()


def build_command(
    *,
    python: str,
    algo: str,
    n_step: int,
    seed: int,
    run_group: str,
    transaction_cost: float,
    offline_data_steps: int,
    n_offline_updates: int | None,
    prefix: str,
) -> str:
    parts = [
        python,
        "-m offline_rl.scripts.run_offline",
        f"--base_config={algo}",
        f"--n_step={n_step}",
        f"--seed={seed}",
        f"--run_group={shlex.quote(run_group)}",
        f"--transaction_cost={transaction_cost}",
        f"--offline_data_steps={offline_data_steps}",
    ]
    if n_offline_updates is not None:
        parts.append(f"--n_offline_updates={n_offline_updates}")
    cmd = " ".join(parts)
    return f"{prefix} {cmd}".strip() if prefix else cmd


def main() -> int:
    args = parse_args()
    combos = list(itertools.product(args.algorithms, args.n_steps, args.seeds))

    header = (
        f"# Offline-RL matrix: {len(args.algorithms)} algos × "
        f"{len(args.n_steps)} n_step × {len(args.seeds)} seeds "
        f"= {len(combos)} runs\n"
        f"# transaction_cost={args.transaction_cost}  run_group={args.run_group}\n"
        f"# NOTE: this script only PRINTS commands — it never executes them.\n"
    )
    sys.stdout.write(header)
    for algo, n_step, seed in combos:
        cmd = build_command(
            python=args.python,
            algo=algo,
            n_step=n_step,
            seed=seed,
            run_group=args.run_group,
            transaction_cost=args.transaction_cost,
            offline_data_steps=args.offline_data_steps,
            n_offline_updates=args.n_offline_updates,
            prefix=args.prefix,
        )
        sys.stdout.write(cmd + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
