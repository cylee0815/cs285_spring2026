"""
Training entry point for the O2O pipeline.

Chronological split (aligned to proposal's distribution-shift experiment):
  Train  (offline):  2008-01-01 → 2020-12-31  (GFC 2008 + COVID 2020)
  Val    (HP tuning): 2021-01-01 → 2021-12-31
  Test   (O2O online): 2022-01-01 → 2026-03-31  (stock/bond correlation break)

Usage:
    # Standard ETF universe (SPY, EEM, TLT, HYG, DBC, GLD, UUP, SHY — data from 2008)
    uv run src/scripts/run_o2o.py --run_group=debug --phase=offline --seed=0
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=o2o --seed=0

    # Mutual fund proxies — extends history to 1990s (Dot-Com bubble coverage)
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=o2o --use_mutual_funds --start_date=1995-01-01 --seed=0

    # FinRL env for online phase (richer features: MACD, RSI, CCI, turbulence)
    # Note: offline phase always uses custom env (FinRL env resets from start of data,
    #       not suitable for random-start offline trajectory generation)
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=sac --use_finrl --seed=0
    uv run src/scripts/run_o2o.py --run_group=exp1 --phase=o2o --use_finrl_online --seed=0
"""
import argparse
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from tqdm import trange

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.envs.data_utils import (
    make_train_test_envs, make_train_val_test_envs, make_train_test_envs_finrl,
    DEFAULT_TICKERS, MUTUAL_FUND_TICKERS,
)
from offline_rl.agents.cql_geodesic import GeodesicCQL
from online_rl.agents.sac_dirichlet import SACDirichlet
from hybrid_rl.agents.o2o_agent import O2OAgent
from core.buffers.replay_buffer import ReplayBuffer, NStepReplayBuffer
from policies.behavior import (
    DirichletPolicy, EqualWeightPolicy, MomentumPolicy, RiskParityPolicy,
)
from policies.mixture import default_offline_mixture, make_episode_callable


def _build_behavior(behavior_mix: str, env, seed: int):
    """Return (policy, mixture) compatible with load_from_env. Exactly one is
    non-None; both None means use the legacy uniform-Dirichlet default."""
    if behavior_mix == "mixture":
        return None, default_offline_mixture(env, seed=seed)
    if behavior_mix == "uniform_legacy":
        return None, None
    n_assets = env.action_space.shape[0]
    single_map = {
        "dirichlet": DirichletPolicy(n_assets=n_assets, alpha=1.0, seed=seed),
        "equal_weight": EqualWeightPolicy(n_assets=n_assets),
        "momentum": MomentumPolicy(n_assets=n_assets, lookback=60),
        "risk_parity": RiskParityPolicy(n_assets=n_assets, lookback=60),
    }
    return make_episode_callable(single_map[behavior_mix], env), None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_group", type=str, default="debug")
    parser.add_argument("--phase", type=str, default="o2o",
                        choices=["offline", "online", "o2o", "sac"],
                        help="offline=CQL only, online=SAC only, o2o=full pipeline, sac=online SAC baseline")
    parser.add_argument("--seed", type=int, default=0)
    # Environment — ticker universe
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list (overrides --use_mutual_funds).")
    parser.add_argument(
        "--use_mutual_funds",
        action="store_true",
        help="Use mutual fund proxies (VFINX, VUSTX, etc.) instead of standard ETFs. "
             "Extends history back to the 1990s for Dot-Com bubble coverage.",
    )
    # Chronological split dates
    parser.add_argument("--start_date", type=str, default="2008-01-01",
                        help="Start of training data. Use 1995-01-01 with --use_mutual_funds.")
    parser.add_argument("--train_end",  type=str, default="2020-12-31")
    parser.add_argument("--val_start",  type=str, default="2021-01-01")
    parser.add_argument("--val_end",    type=str, default="2021-12-31")
    parser.add_argument("--test_start", type=str, default="2022-01-01")
    parser.add_argument("--end_date",   type=str, default="2026-03-31",
                        help="End of test data.")
    parser.add_argument("--episode_length", type=int, default=63)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--reward_type", type=str, default="log_return")
    # FinRL options
    parser.add_argument(
        "--use_finrl_online",
        action="store_true",
        help="Use FinRL env for the online fine-tuning phase (SAC/O2O). "
             "Offline phase always uses the custom env regardless of this flag.",
    )
    parser.add_argument(
        "--use_finrl",
        action="store_true",
        help="Shorthand: use FinRL env for online phase (same as --use_finrl_online).",
    )
    parser.add_argument("--finrl_time_window", type=int, default=20)
    # Multimodal feature flags (hypothesis 6: multimodal information advantage)
    parser.add_argument(
        "--use_macro", action="store_true",
        help="Append 8 FRED macroeconomic features (rates, CPI, unemployment, GDP, sentiment). "
             "Requires FRED_API_KEY env var (free key at fred.stlouisfed.org). Falls back to zeros.",
    )
    parser.add_argument(
        "--use_sentiment", action="store_true",
        help="Append SF Fed Daily News Sentiment Index (auto-downloaded on first run).",
    )
    parser.add_argument(
        "--use_alpaca_embeddings", action="store_true",
        help="Append Alpaca News sentence embeddings (384-d). Requires precomputed cache; "
             "see src/envs/sentiment_features.AlpacaNewsEmbeddings.",
    )
    # Training scale
    parser.add_argument("--n_offline_updates", type=int, default=None)
    parser.add_argument("--n_online_steps", type=int, default=None)
    parser.add_argument("--offline_data_steps", type=int, default=50_000)
    parser.add_argument("--eval_interval", type=int, default=5_000)
    # Bayesian regime encoder
    parser.add_argument(
        "--bayesian", action="store_true",
        help="Use BayesianRegimeEncoder for uncertainty-aware regime detection. "
             "Enables Var[h_t]-weighted CQL penalty and Thompson sampling exploration.",
    )
    # Multi-step returns
    parser.add_argument("--n_step", type=int, default=1,
                        help="N-step returns (1=standard, 3/5/10 for multi-step)")
    parser.add_argument(
        "--behavior_mix", type=str, default="mixture",
        choices=["mixture", "dirichlet", "equal_weight", "momentum", "risk_parity",
                 "uniform_legacy"],
        help=(
            "Offline-buffer behavior policy. 'mixture' = canonical 4-way mix; "
            "'uniform_legacy' = pre-Phase-2 hardcoded uniform Dirichlet."
        ),
    )
    parser.add_argument(
        "--adaptive_conservatism", type=str, default="true",
        choices=["true", "false"],
        help=(
            "If 'true' (default), use the regime-KL-driven sigmoid schedule "
            "for cql_weight during online fine-tuning. If 'false', pin "
            "cql_weight = config.cql_alpha throughout — this is the Phase 2C "
            "'naive fine-tune' condition."
        ),
    )
    parser.add_argument("--results_dir", type=str, default=None,
                        help="If set, write metrics.json under {results_dir}/{run_name}.")
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    use_finrl_online = args.use_finrl_online or args.use_finrl

    # ── 3-way chronological split (offline=train, val=HP tuning, online=test) ──
    # Offline phase always uses the custom env (random-start episode sampling required).
    tickers = args.tickers  # None → make_train_val_test_envs selects based on use_mutual_funds
    ticker_label = (
        "mutual_fund_proxies" if (args.use_mutual_funds and tickers is None)
        else str(tickers or DEFAULT_TICKERS)
    )
    print(f"Downloading market data: {ticker_label}")

    custom_train_env, custom_val_env, custom_test_env, metadata = make_train_val_test_envs(
        use_mutual_funds=args.use_mutual_funds,
        tickers=tickers,
        train_start=args.start_date,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.end_date,
        episode_length=args.episode_length,
        transaction_cost=args.transaction_cost,
        reward_type=args.reward_type,
        use_macro=args.use_macro,
        use_sentiment=args.use_sentiment,
        use_alpaca_embeddings=args.use_alpaca_embeddings,
        # Geodesic-CQL (offline) and SAC-Dirichlet (online) both output weights
        # on the simplex via DirichletActor, so the custom env must accept
        # those directly and not apply another softmax.
        accept_portfolio_weights=True,
        fred_api_key=os.environ.get("FRED_API_KEY"),
    )

    # Online phase: optionally use FinRL env for richer observations
    if use_finrl_online and args.phase in ("sac", "o2o", "online"):
        if args.phase == "o2o":
            # O2O transfer requires matching obs_dim between offline (custom) and online envs.
            # FinRL env has a completely different obs_dim (e.g. ~1600 vs ~56), which would
            # crash during weight transfer (regime encoder GRU input dim mismatch).
            print("  WARNING: --use_finrl_online is incompatible with --phase=o2o.")
            print("  The offline phase uses custom env (obs_dim={}) but FinRL env has a"
                  " different obs_dim.".format(metadata['obs_dim']))
            print("  Falling back to custom env for online phase.")
            use_finrl_online = False
            online_train_env, online_test_env = custom_test_env, custom_test_env
            metadata["env_backend"] = "custom"
        else:
            print("  [Online] Using FinRL environment (test split)")
            if args.use_macro:
                print("  WARNING: --use_macro is ignored with FinRL env. "
                      "FinRL uses its own feature pipeline (MACD, RSI, CCI, turbulence).")
            # DirichletActor outputs portfolio weights → accept_portfolio_weights=True
            online_train_env, online_test_env, finrl_meta = make_train_test_envs_finrl(
                tickers=metadata["tickers"],
                start=args.test_start,
                end=args.end_date,
                time_window=args.finrl_time_window,
                transaction_cost=args.transaction_cost,
                accept_portfolio_weights=True,  # DirichletActor: weights → log(w) → softmax → w
            )
            metadata.update({k: v for k, v in finrl_meta.items() if k not in metadata})
            metadata["env_backend"] = "finrl"
            print(f"  [FinRL] obs_dim={finrl_meta['obs_dim']}, action_dim={finrl_meta['action_dim']}")
    else:
        online_train_env, online_test_env = custom_test_env, custom_test_env
        metadata["env_backend"] = "custom"

    train_env = custom_train_env   # offline pre-training (2005–2020)
    val_env   = custom_val_env     # hyperparameter evaluation (2021)
    test_env  = online_test_env    # O2O fine-tuning & final eval (2022+)

    print(f"Train (offline): {metadata['train_start']} → {metadata['train_end']}  ({metadata['T_train']} days)")
    print(f"Val   (HP eval): {metadata['val_start']} → {metadata['val_end']}  ({metadata['T_val']} days)")
    print(f"Test  (online):  {metadata['test_start']} → {metadata['test_end']}  ({metadata['T_test']} days)")

    # Load config
    if args.bayesian and args.phase in ["offline", "o2o"]:
        from hybrid_rl.configs.bayesian_o2o_config import get_config
    elif args.phase in ["offline", "o2o"]:
        from hybrid_rl.configs.o2o_config import get_config
    else:
        from online_rl.configs.sac_dirichlet_config import get_config
    config = get_config()

    # Apply CLI overrides
    if args.n_offline_updates:
        config.n_offline_updates = args.n_offline_updates
    if args.n_online_steps:
        config.n_online_steps = args.n_online_steps
    if args.bayesian:
        config.bayesian = True
    if args.n_step > 1:
        config.n_step = args.n_step
    config.adaptive_conservatism = (args.adaptive_conservatism == "true")

    # Init WandB
    run_name = args.run_name or f"{args.phase}_seed{args.seed}"
    wandb.init(
        project="cs285-portfolio-rl",
        group=args.run_group,
        name=run_name,
        config={**dict(config), **vars(args), **metadata},
    )

    # --- Full O2O pipeline ---
    if args.phase == "o2o":
        # Offline phase uses train env (2005–2020); online phase uses test env (2022+)
        agent = O2OAgent(custom_train_env, test_env, config, device)
        # Swap the SAC agent's env to the online env (FinRL or custom test split)
        agent.sac_agent.env = online_train_env

        # Load offline data
        _bp, _bmix = _build_behavior(args.behavior_mix, train_env, args.seed)
        agent.load_offline_data(
            behavioral_policy=_bp,
            policy_mixture=_bmix,
            n_steps=args.offline_data_steps,
            mixture_seed=args.seed,
        )

        # Phase 1: offline pre-training — evaluate on val split (2021) to avoid look-ahead bias
        print("\n=== Phase 1: Offline Geodesic-CQL Pre-training ===")
        n_offline = config.n_offline_updates
        for step in trange(n_offline, desc="Offline"):
            metrics = agent.cql_agent.update()
            if step % args.eval_interval == 0:
                eval_metrics = agent._evaluate_cql(n_episodes=5)
                wandb.log({**{f"offline/{k}": v for k, v in metrics.items()},
                           **eval_metrics, "step": step})

        # Phase 2: online fine-tuning on test split (2022+).
        # Use ``agent.finetune_online`` so the adaptive (or pinned)
        # ``cql_weight`` schedule actually drives the critic update — the
        # previous inline loop called ``update_critic`` without a cql_weight
        # and silently ran a naive-online fine-tune regardless of the
        # ``--adaptive_conservatism`` flag.
        print("\n=== Phase 2: Online SAC-Dirichlet Fine-tuning ===")
        n_online = config.n_online_steps
        o2o_history = agent.finetune_online(n_online)
        # Final eval on test env after fine-tuning. The mid-training eval the
        # old inline loop produced is dropped intentionally — for Phase 2C the
        # final test metrics + the cql_weight trajectory carry the analysis.
        final_eval = agent.sac_agent.evaluate(test_env, n_episodes=5)
        wandb.log({
            **{f"final_test/{k.split('/')[-1]}": v for k, v in final_eval.items()},
            "step": n_offline + n_online,
        })
        if args.results_dir is not None:
            import json as _json, numpy as _np
            out_dir = os.path.join(args.results_dir, run_name)
            os.makedirs(out_dir, exist_ok=True)
            cql_w_traj = _np.asarray(
                [float(h.get("cql_weight", 0.0)) for h in o2o_history],
                dtype=_np.float32,
            )
            _np.save(os.path.join(out_dir, "cql_weight_traj.npy"), cql_w_traj)
            record = {
                "phase": "o2o",
                "seed": args.seed,
                "transaction_cost": float(args.transaction_cost),
                "adaptive_conservatism": (args.adaptive_conservatism == "true"),
                "n_offline_updates": int(n_offline),
                "n_online_steps": int(n_online),
                "test": {
                    "sharpe_ratio": float(final_eval.get("eval/sharpe_ratio", float("nan"))),
                    "annual_return": float(final_eval.get("eval/annual_return", float("nan"))),
                    "max_drawdown": float(final_eval.get("eval/max_drawdown", float("nan"))),
                    "turnover": float(final_eval.get("eval/avg_turnover", float("nan"))),
                    "cumulative_return": (
                        float(final_eval["eval/portfolio_value"]) - 1.0
                        if "eval/portfolio_value" in final_eval else None
                    ),
                    "episode_return": float(final_eval.get("eval/episode_return", float("nan"))),
                },
                "cql_weight": {
                    "mean": float(cql_w_traj.mean()) if cql_w_traj.size else None,
                    "min": float(cql_w_traj.min()) if cql_w_traj.size else None,
                    "max": float(cql_w_traj.max()) if cql_w_traj.size else None,
                    "n_steps_logged": int(cql_w_traj.size),
                },
            }
            with open(os.path.join(out_dir, "metrics.json"), "w") as _f:
                _json.dump(record, _f, indent=2)
            print(f"[save] {out_dir}/metrics.json")

    # --- Online SAC-Dirichlet only (baseline) ---
    elif args.phase == "sac":
        agent = SACDirichlet(online_train_env, config, device)
        n_steps = config.get("n_online_steps", 200_000)
        for step in trange(n_steps, desc="SAC-Dirichlet"):
            metrics = agent.update()
            if step % args.eval_interval == 0 and metrics:
                eval_metrics = agent.evaluate(test_env, n_episodes=5)
                wandb.log({**{f"train/{k}": v for k, v in metrics.items()},
                           **eval_metrics, "step": step})

    # --- Offline only ---
    elif args.phase == "offline":
        obs_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        if args.n_step > 1:
            offline_buffer = NStepReplayBuffer(
                config.offline_buffer_size, obs_dim, action_dim, device,
                seq_len=config.regime_window, n_step=args.n_step, gamma=config.gamma,
            )
        else:
            offline_buffer = ReplayBuffer(config.offline_buffer_size, obs_dim, action_dim, device,
                                          seq_len=config.regime_window)
        _bp, _bmix = _build_behavior(args.behavior_mix, train_env, args.seed)
        offline_buffer.load_from_env(
            train_env,
            n_steps=args.offline_data_steps,
            policy=_bp,
            policy_mixture=_bmix,
            mixture_seed=args.seed,
        )
        offline_buffer.freeze()
        agent = GeodesicCQL(obs_dim, action_dim, config, device, offline_buffer=offline_buffer)
        n_steps = config.n_offline_updates
        for step in trange(n_steps, desc="Geodesic-CQL"):
            metrics = agent.update()
            if step % args.eval_interval == 0:
                # Evaluate on val split (2021) during offline training
                tmp = O2OAgent.__new__(O2OAgent)
                tmp.eval_env = val_env
                tmp.cql_agent = agent
                tmp.device = device
                eval_metrics = tmp._evaluate_cql(n_episodes=5)
                wandb.log({**{f"offline/{k}": v for k, v in metrics.items()},
                           **eval_metrics, "step": step})

    wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
