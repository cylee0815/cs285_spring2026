"""CLI entrypoint for IQL training.

Usage
-----
    python scripts/train.py --dataset datasets/dirichlet_dataset.npz --steps 100000
    python scripts/train.py --config configs/iql_default.yaml
    python scripts/train.py --config configs/iql_default.yaml --expectile 0.8
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
import yaml

from algorithms.iql import IQL
from training.train_iql import train_iql
from utils.replay_buffer import ReplayBuffer
from utils.seed import resolve_device, set_seed


def _load_yaml_config(path: str) -> dict:
    """Load a YAML config and return as a flat dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IQL on an offline dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to .npz dataset.")
    parser.add_argument("--steps", type=int, default=None, help="Number of gradient steps.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--expectile", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--polyak", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints",
        help="Directory to save model checkpoint.",
    )
    args = parser.parse_args()

    # Build effective config: YAML defaults → CLI overrides
    defaults = {
        "dataset": "datasets/dirichlet_dataset.npz",
        "steps": 100_000,
        "batch_size": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "expectile": 0.7,
        "beta": 3.0,
        "polyak": 0.005,
        "seed": 42,
        "device": "auto",
        "log_interval": 1000,
    }
    if args.config is not None:
        yaml_cfg = _load_yaml_config(args.config)
        defaults.update(yaml_cfg)
    # CLI args override YAML
    for key in ["dataset", "steps", "batch_size", "lr", "gamma", "expectile",
                "beta", "polyak", "seed", "device", "log_interval"]:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            defaults[key] = cli_val

    cfg = argparse.Namespace(**defaults)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Device: {device}")

    # Infer dims from dataset
    data = np.load(cfg.dataset)
    state_dim = data["states"].shape[1]
    action_dim = data["actions"].shape[1]
    print(f"Dataset: {data['states'].shape[0]} transitions, state_dim={state_dim}, action_dim={action_dim}")

    buffer = ReplayBuffer(cfg.dataset, device=device)
    agent = IQL(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=cfg.lr,
        gamma=cfg.gamma,
        tau=cfg.expectile,
        beta=cfg.beta,
        polyak=cfg.polyak,
        device=device,
    )

    train_iql(
        agent=agent,
        buffer=buffer,
        total_steps=cfg.steps,
        batch_size=cfg.batch_size,
        log_interval=cfg.log_interval,
    )

    # Save checkpoint
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "iql.pt"
    import torch
    torch.save({
        "state_dim": state_dim,
        "action_dim": action_dim,
        "q_network": agent.q_network.state_dict(),
        "value_network": agent.value_network.state_dict(),
        "policy_network": agent.policy_network.state_dict(),
        "gamma": cfg.gamma,
        "tau": cfg.expectile,
        "beta": cfg.beta,
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
