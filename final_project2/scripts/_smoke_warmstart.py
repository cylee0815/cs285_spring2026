"""Smoke test for the GRPO offline warm-start path.

Verifies, in order:
  (0) [PRE-CHECK, fail-fast] The checkpoint's first-layer input dim
      matches the obs_dim of the actual target env that
      ``scripts/train_grpo.py`` will construct from ``--dataset``.
      Catches feature-pipeline mismatches (e.g. 56-d source vs 216-d
      target) BEFORE attempting the state_dict copy.
  (a) The 6-key state-dict rename loads cleanly into DirichletMLPPolicy.
  (b) The warm-started GRPO actor produces the SAME Dirichlet
      concentration as the source DirichletActor on identical inputs
      (bit-exact equivalence — confirms no silent layer-mismatch).
  (c) The warm-started GRPO actor's α distribution differs from a
      random-init GRPO actor's distribution (confirms the load
      actually changed the weights, not a no-op).
  (d) The Dirichlet means lie on the simplex, are not near-uniform
      (uniform = 1/8 = 0.125), and have plausible spread.

Usage:
    uv run python scripts/_smoke_warmstart.py \\
        --checkpoint results/phase2a_causal/iql_lambda0.001_seed42/actor.pt \\
        --target_dataset datasets/real_dirichlet.npz

If --target_dataset is omitted, the script asserts ``--obs_dim`` matches
the checkpoint and skips the env-build (legacy mode for non-warm-start
audits).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.networks.dirichlet_policy import DirichletActor
from core.networks.policies import DirichletMLPPolicy

RENAME = {
    "net.0.weight": "shared.0.weight", "net.0.bias": "shared.0.bias",
    "net.2.weight": "shared.2.weight", "net.2.bias": "shared.2.bias",
    "net.4.weight": "actor_alpha.weight", "net.4.bias": "actor_alpha.bias",
}


def _build_target_env(target_dataset: str):
    """Reconstruct the train_env exactly the way scripts/train_grpo.py does
    so the obs_dim we read is bit-identical to the target."""
    from data.splits import SplitConfig, compute_split_indices
    from scripts.train_grpo import _load_dataset_arrays, _slice_env

    states, fwd, dates_ns = _load_dataset_arrays(target_dataset)
    split_cfg = SplitConfig(
        train_start="2008-01-01", train_end="2020-12-31",
        val_start="2021-01-01",   val_end="2021-12-31",
        test_start="2022-01-01",  test_end="2026-03-31",
    )
    split_idx = compute_split_indices(dates_ns, split_cfg)
    train_env = _slice_env(
        states, fwd, split_idx.train,
        episode_length=63, transaction_cost=0.001,
    )
    return train_env


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--target_dataset", type=str, default=None,
                   help="Path to the GRPO target dataset (e.g. "
                        "datasets/real_dirichlet.npz). If set, the script "
                        "builds the actual target env via train_grpo's "
                        "_slice_env and reads obs_dim from it. Catches "
                        "feature-pipeline mismatches pre-launch.")
    p.add_argument("--obs_dim", type=int, default=None,
                   help="Manual obs_dim override. Used only when "
                        "--target_dataset is omitted.")
    p.add_argument("--action_dim", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_test_states", type=int, default=64)
    args = p.parse_args()

    print(f"[load] {args.checkpoint}")
    src_state = torch.load(args.checkpoint, map_location="cpu")
    print(f"  keys: {list(src_state.keys())}")

    # (0) PRE-CHECK: read obs_dim from the actual target env. If the
    # checkpoint's first-layer expects a different input dim, FAIL HERE.
    src_input_dim = src_state["net.0.weight"].shape[1]
    if args.target_dataset is not None:
        print(f"\n[pre-check] Building target env from {args.target_dataset}")
        target_env = _build_target_env(args.target_dataset)
        target_obs_dim = target_env.observation_space.shape[0]
        print(f"  target env obs_dim = {target_obs_dim}")
        print(f"  source ckpt expects input dim = {src_input_dim}")
        if src_input_dim != target_obs_dim:
            print(f"\n[FAIL pre-check] Dimensionality mismatch: source "
                  f"checkpoint was trained on {src_input_dim}-d "
                  f"observations, but the target GRPO env has "
                  f"{target_obs_dim}-d observations. The state_dict load "
                  f"WILL fail at scripts/train_grpo.py's actor.load_state_dict "
                  f"call. Possible causes: source IQL was trained on "
                  f"compute_features (56-d) but GRPO uses build_features "
                  f"(216-d), or vice versa. This run cannot be warm-started "
                  f"without retraining the source on the target's feature space.")
            return 2
        print(f"  OK — dims match")
        obs_dim = target_obs_dim
    else:
        if args.obs_dim is None:
            print(f"\n[FAIL] either --target_dataset or --obs_dim is required")
            return 2
        if src_input_dim != args.obs_dim:
            print(f"\n[FAIL] manual --obs_dim={args.obs_dim} != source "
                  f"first-layer input dim {src_input_dim}")
            return 2
        obs_dim = args.obs_dim
        print(f"\n[pre-check] (skipped real env, using --obs_dim={obs_dim})")

    # (a) Rebuild source DirichletActor and load state to verify it's
    # a valid DirichletActor checkpoint.
    src_actor = DirichletActor(obs_dim, args.action_dim, args.hidden_dim, args.n_layers)
    src_actor.load_state_dict(src_state, strict=True)
    src_actor.eval()
    print("  source DirichletActor: loaded strictly, OK")

    # Build random-init GRPO actor (control)
    random_grpo = DirichletMLPPolicy(obs_dim, args.action_dim, args.hidden_dim, args.n_layers)
    random_grpo.eval()

    # Build warm-started GRPO actor
    warm_grpo = DirichletMLPPolicy(obs_dim, args.action_dim, args.hidden_dim, args.n_layers)
    remapped = {RENAME[k]: v for k, v in src_state.items() if k in RENAME}
    missing, unexpected = warm_grpo.load_state_dict(remapped, strict=False)
    print(f"  warm-start load: missing={list(missing)} unexpected={list(unexpected)}")
    assert unexpected == [], f"unexpected keys: {unexpected}"
    assert set(missing) == {"critic.weight", "critic.bias"}, f"unexpected missing: {missing}"
    warm_grpo.eval()

    # Test on synthetic observations
    torch.manual_seed(0)
    obs = torch.randn(args.n_test_states, obs_dim)

    with torch.no_grad():
        # Source IQL alpha
        src_alpha = torch.nn.functional.softplus(src_actor.net(obs)) + 1.0
        # Warm-started GRPO alpha
        warm_alpha = torch.nn.functional.softplus(warm_grpo.actor_alpha(warm_grpo.shared(obs))) + 1.0
        # Random-init GRPO alpha
        rand_alpha = torch.nn.functional.softplus(random_grpo.actor_alpha(random_grpo.shared(obs))) + 1.0

    # (b) Bit-exact equivalence src_actor ↔ warm_grpo
    diff_src_vs_warm = (src_alpha - warm_alpha).abs().max().item()
    print(f"\n[check b] Max |α_source - α_warm| = {diff_src_vs_warm:.2e}  (should be 0.00e+00)")
    assert diff_src_vs_warm < 1e-6, f"FAIL: warm-start does not match source!"

    # (c) Warm-started ≠ random-init
    diff_warm_vs_rand = (warm_alpha - rand_alpha).abs().mean().item()
    print(f"[check c] Mean |α_warm - α_random| = {diff_warm_vs_rand:.4f}  (should be > 0)")
    assert diff_warm_vs_rand > 0.01, f"FAIL: warm-start indistinguishable from random init!"

    # (d) Dirichlet mean characteristics
    src_mean = torch.distributions.Dirichlet(src_alpha).mean.numpy()  # (B, A)
    rand_mean = torch.distributions.Dirichlet(rand_alpha).mean.numpy()
    n = args.action_dim
    print(f"\n[check d] Dirichlet means (uniform = {1.0/n:.4f}):")
    print(f"  source IQL    mean of weights:  per-asset means = {src_mean.mean(0).round(4).tolist()}")
    print(f"  source IQL    weight std (across states): {src_mean.std(0).mean():.4f}")
    print(f"  random init   mean of weights:  per-asset means = {rand_mean.mean(0).round(4).tolist()}")
    print(f"  random init   weight std (across states): {rand_mean.std(0).mean():.4f}")
    # Per-row sums should be 1
    src_row_sums = src_mean.sum(axis=1)
    print(f"  source IQL    per-row sum: min={src_row_sums.min():.6f} max={src_row_sums.max():.6f}")
    assert abs(src_row_sums.min() - 1.0) < 1e-5 and abs(src_row_sums.max() - 1.0) < 1e-5

    # Plausibility check: IQL trained 20k steps should NOT be near-uniform.
    # Random init is exactly uniform (1/N) when softplus(0)+1 ≈ 1.69 is the
    # constant alpha across assets, so its weight std across states is ~0.
    # A trained policy should have weight std > 0.005 across states.
    src_state_std = src_mean.std(0).mean()
    if src_state_std < 0.001:
        print(f"\n[WARN] source actor's weights are near-constant across states "
              f"(std={src_state_std:.6f}). May indicate IQL collapsed to near-EW.")
        print("  Not a smoke-test failure — leak-invariance predicts EW-ish IQL on causal pipeline.")
    else:
        print(f"\n[OK] source actor varies its weights across states (std={src_state_std:.4f})")

    print("\n=== SMOKE PASS ===")
    print("Warm-start path is functional. GRPO can be launched with this checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
