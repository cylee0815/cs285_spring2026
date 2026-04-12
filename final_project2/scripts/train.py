"""CLI entrypoint for IQL training.

Usage
-----
    python scripts/train.py --dataset datasets/dirichlet_dataset.npz --steps 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so bare imports work.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from algorithms.iql import IQL
from training.train_iql import train_iql
from utils.replay_buffer import ReplayBuffer
from utils.seed import resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IQL on an offline dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset.")
    parser.add_argument("--steps", type=int, default=100_000, help="Number of gradient steps.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Infer dims from dataset
    data = np.load(args.dataset)
    state_dim = data["states"].shape[1]
    action_dim = data["actions"].shape[1]
    print(f"Dataset: {data['states'].shape[0]} transitions, state_dim={state_dim}, action_dim={action_dim}")

    buffer = ReplayBuffer(args.dataset, device=device)
    agent = IQL(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.expectile,
        beta=args.beta,
        polyak=args.polyak,
        device=device,
    )

    train_iql(
        agent=agent,
        buffer=buffer,
        total_steps=args.steps,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
